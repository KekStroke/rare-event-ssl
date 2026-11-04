"""Strict, local-only encoder loading and single-record physical feature grids.

AVES receives one unpadded waveform at a time. Only audio shorter than the
convolutional receptive field gets explicit minimum-input padding; that fact is
returned with the features. No evaluation batching or five-second padding occurs.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class EncoderSpec:
    kind: str
    checkpoint: str
    config: str | None = None
    expected_sha256: str | None = None
    config_sha256: str | None = None
    preprocessor_sha256: str | None = None
    revision: str | None = None
    model_id: str | None = None
    normalize_waveform: bool | None = None


@dataclass(frozen=True)
class FrameMetadata:
    sample_rate: int
    original_samples: int
    model_input_samples: int
    padding_samples: int
    feature_frames: int
    fully_unpadded_frames: int
    hop_samples: int
    conv_receptive_field_samples: int
    center_offset_seconds: float
    frame_hop_seconds: float
    source_duration_seconds: float
    waveform_normalized: bool


@dataclass(frozen=True)
class EncoderOutput:
    layers: dict[int, Any]
    metadata: FrameMetadata


def _verify_file(path, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
        raise ValueError("A registered source SHA-256 is required before model loading")
    path = Path(path)
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != expected:
        raise ValueError(f"Registered artifact SHA-256 mismatch: {path.name}")
    return path


def _conv_geometry(kernels, strides):
    if len(kernels) != len(strides) or not kernels:
        raise ValueError("Convolution kernels/strides must have equal nonzero length")
    receptive, hop = 1, 1
    for kernel, stride in zip(kernels, strides):
        if not isinstance(kernel, int) or not isinstance(stride, int) or min(kernel, stride) < 1:
            raise ValueError("Invalid convolution geometry")
        receptive += (kernel - 1) * hop
        hop *= stride
    return receptive, hop


def _frame_count(samples, receptive, hop):
    return max(0, (samples - receptive) // hop + 1)


def _strict_hubert_state(model, state):
    """Migrate only PyTorch's documented positional-convolution weight_norm pair.

    PyTorch 2.2's compatibility hook maps g/v to original0/original1 in the
    same order, without changing either tensor. Validate full coverage before
    strict loading rather than accepting Transformers' partial-load warnings.
    https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/utils/parametrizations.py
    """
    import torch
    if not isinstance(state, Mapping) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a plain HF tensor state dictionary")
    expected = model.state_dict()
    mapped, migrations = dict(state), []
    prefix = "encoder.pos_conv_embed.conv."
    pairs = [(prefix + "weight_g", prefix + "parametrizations.weight.original0"),
             (prefix + "weight_v", prefix + "parametrizations.weight.original1")]
    if any(old in mapped for old, _ in pairs) and any(new in expected for _, new in pairs):
        if not all(old in mapped and new in expected and new not in mapped for old, new in pairs):
            raise ValueError("Incomplete or ambiguous positional-convolution weight_norm state")
        for old, new in pairs:
            mapped[new] = mapped.pop(old)
            migrations.append({"from": old, "to": new})
    missing, unexpected = sorted(set(expected) - set(mapped)), sorted(set(mapped) - set(expected))
    if missing or unexpected:
        raise ValueError("HF encoder key coverage mismatch: " + json.dumps({"missing": missing, "unexpected": unexpected}))
    for key, tensor in mapped.items():
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise ValueError("HF encoder tensor shape/dtype mismatch: " + key)
    model.load_state_dict(mapped, strict=True)
    return {"strict": True, "checkpoint_tensors": len(state), "model_tensors": len(expected),
            "weight_norm_key_migrations": migrations, "missing_keys": [], "unexpected_keys": []}


class EncoderAdapter:
    def __init__(self, model, *, kind, device, kernels, strides, num_layers,
                 embedding_dim, normalize_waveform=False):
        import torch
        self.model = model.eval().to(device)
        self.model.requires_grad_(False)
        self.kind, self.device = kind, torch.device(device)
        self.receptive, self.hop = _conv_geometry(kernels, strides)
        self.num_layers, self.embedding_dim = num_layers, embedding_dim
        self.normalize_waveform = bool(normalize_waveform)

    def extract_layers(self, wave1d, sample_rate=16000, layers=(6, 12)):
        import torch
        import torch.nn.functional as functional
        if sample_rate != 16000:
            raise ValueError("Encoder input must be physically sampled at 16000 Hz")
        wave = torch.as_tensor(wave1d)
        if wave.ndim != 1 or wave.numel() == 0 or not wave.is_floating_point():
            raise ValueError("Expected one nonempty floating-point mono waveform; batching is disabled")
        ids = list(layers)
        if (not ids or len(set(ids)) != len(ids)
                or any(not isinstance(layer, int) or isinstance(layer, bool)
                       or not 1 <= layer <= self.num_layers for layer in ids)):
            raise ValueError("Transformer layer IDs must be unique and one-based")
        wave = wave.detach().to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(wave).all()):
            raise ValueError("Nonfinite waveform")
        original = wave.numel()
        # Match the pinned HF feature extractor on the original, unpadded audio.
        if self.normalize_waveform:
            wave = (wave - wave.mean()) / torch.sqrt(wave.var(unbiased=False) + 1e-7)
        padding = max(0, self.receptive - original)
        if padding:
            wave = functional.pad(wave, (0, padding))
        expected = _frame_count(wave.numel(), self.receptive, self.hop)
        with torch.inference_mode():
            if self.kind == "aves_torchaudio":
                values, _ = self.model.extract_features(wave.unsqueeze(0), num_layers=max(ids))
                selected = {layer: values[layer - 1][0] for layer in ids}
            elif self.kind == "hubert_transformers":
                values = self.model(wave.unsqueeze(0), output_hidden_states=True, return_dict=True).hidden_states
                selected = {layer: values[layer][0] for layer in ids}
            else:
                raise ValueError("Unsupported encoder backend")
        for value in selected.values():
            if tuple(value.shape) != (expected, self.embedding_dim) or not bool(torch.isfinite(value).all()):
                raise ValueError("Encoder produced unexpected frame/dimension count or nonfinite features")
        return EncoderOutput(selected, FrameMetadata(
            sample_rate=sample_rate, original_samples=original, model_input_samples=wave.numel(),
            padding_samples=padding, feature_frames=expected,
            fully_unpadded_frames=_frame_count(original, self.receptive, self.hop),
            hop_samples=self.hop, conv_receptive_field_samples=self.receptive,
            center_offset_seconds=(self.receptive - 1) / (2 * sample_rate),
            frame_hop_seconds=self.hop / sample_rate, source_duration_seconds=original / sample_rate,
            waveform_normalized=self.normalize_waveform))


def load_encoder(spec: EncoderSpec | Mapping, device="cpu"):
    """Load only registered local assets; never download or reinterpret legacy weights."""
    if isinstance(spec, Mapping):
        spec = EncoderSpec(**spec)
    if spec.kind not in {"aves_torchaudio", "hubert_transformers"}:
        raise ValueError("Legacy Fairseq checkpoints require a separately verified strict backbone adapter")
    if spec.revision is not None and not re.fullmatch(r"[0-9a-f]{40}", spec.revision):
        raise ValueError("Model revision must be an immutable commit SHA")
    import torch
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is disabled")
    if spec.kind == "aves_torchaudio":
        from torchaudio.models import wav2vec2_model
        weight = _verify_file(spec.checkpoint, spec.expected_sha256)
        configuration = _verify_file(spec.config, spec.config_sha256)
        config = json.loads(configuration.read_text(encoding="utf-8"))
        if config.get("aux_num_out") is not None:
            raise ValueError("Published AVES must be loaded as an encoder without an auxiliary head")
        config["aux_num_out"] = None
        model = wav2vec2_model(**config)
        state = torch.load(weight, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected the published TorchAudio tensor state dictionary; no implicit key rewriting")
        model.load_state_dict(state, strict=True)
        conv = config["extractor_conv_layer_config"]
        return EncoderAdapter(model, kind=spec.kind, device=target,
                              kernels=[row[1] for row in conv], strides=[row[2] for row in conv],
                              num_layers=config["encoder_num_layers"], embedding_dim=config["encoder_embed_dim"],
                              normalize_waveform=spec.normalize_waveform or False)
    from transformers import HubertConfig, HubertModel
    directory = Path(spec.checkpoint)
    _verify_file(directory / "pytorch_model.bin", spec.expected_sha256)
    _verify_file(directory / "config.json", spec.config_sha256)
    preprocessing = _verify_file(directory / "preprocessor_config.json", spec.preprocessor_sha256)
    processor = json.loads(preprocessing.read_text(encoding="utf-8"))
    if processor.get("sampling_rate") != 16000 or not isinstance(processor.get("do_normalize"), bool):
        raise ValueError("Pinned HF preprocessing does not describe the expected 16 kHz input")
    model = HubertModel(HubertConfig.from_json_file(str(directory / "config.json")))
    state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    loading = _strict_hubert_state(model, state)
    normalization = processor["do_normalize"] if spec.normalize_waveform is None else spec.normalize_waveform
    adapter = EncoderAdapter(model, kind=spec.kind, device=target,
                             kernels=list(model.config.conv_kernel), strides=list(model.config.conv_stride),
                             num_layers=model.config.num_hidden_layers, embedding_dim=model.config.hidden_size,
                             normalize_waveform=normalization)
    adapter.loading_report = loading
    return adapter
