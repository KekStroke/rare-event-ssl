import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from marine_ssl.experiments.encoders import (
    EncoderAdapter, EncoderSpec, _conv_geometry, _strict_hubert_state, _verify_file, load_encoder,
)

TORCH = importlib.util.find_spec("torch") is not None
KERNELS = [10, 3, 3, 3, 3, 2, 2]
STRIDES = [5, 2, 2, 2, 2, 2, 2]


class EncoderIntegrityTests(unittest.TestCase):
    def test_physical_conv_geometry_and_rejection(self):
        self.assertEqual((400, 320), _conv_geometry(KERNELS, STRIDES))
        with self.assertRaises(ValueError):
            _conv_geometry([10], [0])
        with self.assertRaises(ValueError):
            load_encoder(EncoderSpec("unverified_fairseq", "unused"))

    def test_registered_hash_is_required_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.write_bytes(b"registered")
            checksum = hashlib.sha256(b"registered").hexdigest()
            self.assertEqual(path, _verify_file(path, checksum))
            with self.assertRaises(ValueError):
                _verify_file(path, None)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                _verify_file(path, checksum)


@unittest.skipUnless(TORCH, "Torch is installed in the GPU smoke environment")
class SingleRecordFeatureTests(unittest.TestCase):
    def test_weight_norm_migration_preserves_real_convolution_outputs(self):
        import torch

        def model(modern):
            result = torch.nn.Module()
            result.encoder = torch.nn.Module()
            result.encoder.pos_conv_embed = torch.nn.Module()
            convolution = torch.nn.Conv1d(4, 4, 3, padding=1)
            if modern:
                convolution = torch.nn.utils.parametrizations.weight_norm(convolution, dim=2)
            else:
                convolution = torch.nn.utils.weight_norm(convolution, dim=2)
            result.encoder.pos_conv_embed.conv = convolution
            return result

        old, new = model(False), model(True)
        report = _strict_hubert_state(new, old.state_dict())
        wave = torch.arange(68, dtype=torch.float32).reshape(1, 4, 17) / 100
        torch.testing.assert_close(old.encoder.pos_conv_embed.conv(wave),
                                   new.encoder.pos_conv_embed.conv(wave), rtol=0, atol=0)
        self.assertEqual(2, len(report["weight_norm_key_migrations"]))
        self.assertEqual([], report["missing_keys"])
        malformed = dict(old.state_dict())
        malformed["unrelated.weight"] = torch.ones(1)
        with self.assertRaises(ValueError):
            _strict_hubert_state(new, malformed)
        malformed = dict(old.state_dict())
        malformed["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = torch.ones(1)
        with self.assertRaises(ValueError):
            _strict_hubert_state(new, malformed)

    def adapter(self):
        import torch

        class Model(torch.nn.Module):
            def extract_features(self, wave, num_layers):
                self.last_input = wave.clone()
                frames = (wave.shape[-1] - 400) // 320 + 1
                return [torch.full((1, frames, 4), float(i + 1)) for i in range(num_layers)], None

        return EncoderAdapter(Model(), kind="aves_torchaudio", device="cpu", kernels=KERNELS,
                              strides=STRIDES, num_layers=12, embedding_dim=4)

    def test_unpadded_five_seconds_layer_ids_and_physical_grid(self):
        import torch
        adapter = self.adapter()
        output = adapter.extract_layers(torch.ones(80000), layers=[1, 12])
        self.assertEqual((249, 4), tuple(output.layers[12].shape))
        self.assertTrue(bool((output.layers[1] == 1).all()))
        self.assertTrue(bool((output.layers[12] == 12).all()))
        self.assertEqual(0, output.metadata.padding_samples)
        self.assertEqual(80000, adapter.model.last_input.shape[-1])
        self.assertEqual(249, output.metadata.fully_unpadded_frames)
        self.assertEqual(0.02, output.metadata.frame_hop_seconds)

    def test_only_minimum_convolution_padding_for_short_audio(self):
        import torch
        output = self.adapter().extract_layers(torch.ones(63), layers=[6])
        self.assertEqual(337, output.metadata.padding_samples)
        self.assertEqual(400, output.metadata.model_input_samples)
        self.assertEqual(0, output.metadata.fully_unpadded_frames)
        self.assertEqual((1, 4), tuple(output.layers[6].shape))

    def test_batch_wrong_rate_nonfinite_and_zero_based_layers_rejected(self):
        import torch
        adapter = self.adapter()
        for waveform, rate, layers in ((torch.ones(1, 80000), 16000, [6]),
                                      (torch.ones(80000), 32000, [6]),
                                      (torch.full((800,), float("nan")), 16000, [6]),
                                      (torch.ones(80000), 16000, [0])):
            with self.assertRaises(ValueError):
                adapter.extract_layers(waveform, sample_rate=rate, layers=layers)


if __name__ == "__main__":
    unittest.main()
