import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext

from marine_ssl.experiments.pretrain import learning_rate


HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "torchaudio", "numpy"))


class ScheduleTests(unittest.TestCase):
    def test_warmup_polynomial_and_endpoint(self):
        self.assertEqual(learning_rate(0, 10000), 0)
        self.assertAlmostEqual(learning_rate(2500, 10000), 2.5e-5)
        self.assertEqual(learning_rate(5000, 10000), 5e-5)
        self.assertAlmostEqual(learning_rate(7500, 10000), 2.5e-5)
        self.assertEqual(learning_rate(10000, 10000), 0)
        self.assertEqual(learning_rate(20000, 10000), 0)

    def test_explicit_resume_receipt_pins_original_config_and_actual_sources(self):
        from marine_ssl.experiments.pretrain import (AMP_COMPATIBLE_PRETRAIN_SHA256, AMP_RETRY_POLICY,
            implementation_sources, validate_resume_implementation)
        from marine_ssl.experiments.codebook import digest, fingerprint
        actual = implementation_sources()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = {**actual, "pretrain.py": AMP_COMPATIBLE_PRETRAIN_SHA256}
            config = {"output_dir": str(root), "training_source_sha256": original, "epochs": 10}
            config_path = root / "original.json"; config_path.write_text(json.dumps(config))
            (root / "latest.json").write_text(json.dumps({"signature": "original-run"}))
            receipt = {"schema_version": 1, "policy_id": AMP_RETRY_POLICY,
                       "config_sha256": digest(config_path), "recipe_fingerprint": fingerprint(config),
                       "run_signature": "original-run", "original_training_source_sha256": original,
                       "actual_training_source_sha256": actual}
            path = root / "implementation.json"; path.write_text(json.dumps(receipt))
            self.assertEqual(validate_resume_implementation(config, path, resume="auto", config_path=config_path)
                             ["actual_training_source_sha256"], actual)
            with self.assertRaisesRegex(ValueError, "explicit resume"):
                validate_resume_implementation(config, path, resume=None, config_path=config_path)
            for changed in ({"actual_training_source_sha256": {**actual, "encoders.py": "changed"}},
                            {"actual_training_source_sha256": {**actual, "pretrain.py": "changed"}},
                            {"run_signature": "other-run"}, {"recipe_fingerprint": "other-recipe"}):
                path.write_text(json.dumps({**receipt, **changed}))
                with self.assertRaises(ValueError):
                    validate_resume_implementation(config, path, resume="auto", config_path=config_path)
            path.write_text(json.dumps(receipt)); config_path.write_text(json.dumps(config) + "\n")
            with self.assertRaisesRegex(ValueError, "config/source"):
                validate_resume_implementation(config, path, resume="auto", config_path=config_path)


@unittest.skipUnless(HAS_STACK, "PyTorch/TorchAudio/NumPy required")
class PretrainStackTests(unittest.TestCase):
    def test_amp_overflow_replays_same_group_and_rng_before_single_commit(self):
        import random
        import numpy as np
        import torch
        from marine_ssl.experiments.pretrain import backward_group, seed_everything

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.3)); self.draws = []
            def forward(self, waveform, labels):
                draw = (random.random(), float(np.random.random()), float(torch.rand(())))
                self.draws.append(draw)
                value = self.weight * (waveform.mean() + sum(draw))
                return torch.stack((value, -value)).reshape(1, 2), None, value.square() * .01

        # CPU fault injection models GradScaler's recorded found_inf/backoff;
        # real AdamW/model/RNG still run, including replay and parameter equality.
        class Scaler:
            def __init__(self, overflows):
                self.overflows = overflows; self.value = 8.; self.tracker = 7; self.commits = 0
            def is_enabled(self): return True
            def get_scale(self): return self.value
            def scale(self, loss): return loss * self.value
            def unscale_(self, optimizer):
                self.found_inf = self.overflows > 0
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        p.grad.div_(self.value)
                        if self.found_inf: p.grad.fill_(float("inf"))
                if self.found_inf: self.overflows -= 1
            def step(self, optimizer):
                if not self.found_inf: optimizer.step(); self.commits += 1
            def update(self):
                if self.found_inf: self.value /= 2; self.tracker = 0
                else: self.tracker += 1

        batches = [{"waveform": torch.ones(1, 4), "labels": torch.zeros(1, 1, dtype=torch.long)}]
        results = []
        for overflows in (0, 1):
            seed_everything(43)
            model, scaler = Model(), Scaler(overflows)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            with patch("marine_ssl.experiments.pretrain._autocast", return_value=nullcontext()):
                result = backward_group(model, optimizer, scaler, batches, device="cpu", precision="fp16")
            self.assertEqual(scaler.commits, 0)
            if overflows:
                self.assertEqual(model.draws[0], model.draws[1])
                self.assertEqual(scaler.tracker, 0); self.assertEqual(scaler.get_scale(), 4)
            scaler.step(optimizer); scaler.update()
            self.assertEqual(scaler.commits, 1)
            results.append((model.weight.detach().clone(), random.random(), float(np.random.random()),
                            float(torch.rand(())), result[3]))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertEqual(results[0][1:4], results[1][1:4]); self.assertEqual(results[1][4], 1)

        model, scaler = Model(), Scaler(20)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        before = model.weight.detach().clone()
        with patch("marine_ssl.experiments.pretrain._autocast", return_value=nullcontext()):
            with self.assertRaisesRegex(FloatingPointError, "bounded same-batch"):
                backward_group(model, optimizer, scaler, batches, device="cpu", precision="fp16", max_retries=2)
        self.assertEqual(len(model.draws), 3); self.assertEqual(scaler.commits, 0)
        self.assertTrue(torch.equal(before, model.weight)); self.assertEqual(scaler.tracker, 0)

        model, scaler = Model(), Scaler(0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        with patch("marine_ssl.experiments.pretrain._autocast", return_value=nullcontext()), \
             patch("marine_ssl.experiments.pretrain.objective_sum", return_value=(model.weight * float("nan"), 1)):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite HuBERT objective"):
                backward_group(model, optimizer, scaler, batches, device="cpu", precision="fp16")
        self.assertEqual(scaler.commits, 0)

    def test_cpu_telemetry_does_not_query_cuda(self):
        from marine_ssl.experiments.pretrain import device_telemetry
        with patch("torch.cuda.get_device_properties", side_effect=AssertionError("CUDA queried")), \
             patch("torch.cuda.current_device", side_effect=AssertionError("CUDA initialized")):
            result = device_telemetry("cpu")
        self.assertEqual(result["device_type"], "cpu")
        self.assertIsNone(result["gpu_peak_allocated_bytes"])
        self.assertIsNone(result["gpu_peak_reserved_bytes"])

    def test_optimizer_decouples_decay_from_gradient_moments(self):
        import torch
        from marine_ssl.experiments.pretrain import make_optimizer
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = make_optimizer([parameter])
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        self.assertAlmostEqual(float(parameter), 2.0 * (1 - 5e-5 * .01), places=6)
        self.assertEqual(float(optimizer.state[parameter]["exp_avg"]), 0)
        self.assertEqual(float(optimizer.state[parameter]["exp_avg_sq"]), 0)

    def test_official_nce_temperature_duplicate_mask_and_objective(self):
        import torch
        from torchaudio.models.wav2vec2.components import _compute_logits
        from marine_ssl.experiments.pretrain import objective_sum, prepare_waveforms
        logits = _compute_logits(torch.tensor([[1., 0.]]), torch.tensor([0]), torch.tensor([[1., 0.], [0., 1.]]))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertAlmostEqual(float(logits[0, 0]), 10)
        self.assertTrue(bool(torch.isneginf(logits[0, 1])))
        self.assertEqual(float(logits[0, 2]), 0)
        loss, count = objective_sum(logits, torch.tensor(.25))
        self.assertAlmostEqual(float(loss / count), 2.5 + float(torch.nn.functional.softplus(torch.tensor(-10.))), places=6)
        model = type("Policy", (), {"input_normalize_waveform": True})()
        values = prepare_waveforms(torch.tensor([[1., 2., 3.], [10., 20., 30.]]), model)
        self.assertTrue(torch.allclose(values.mean(-1), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(values[0], values[1], atol=1e-6))

    def test_aggregate_masked_frame_gradient_normalization(self):
        import torch
        from marine_ssl.experiments.pretrain import objective_sum
        first = torch.tensor(.4, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        pieces = [torch.tensor([1., 2.]), torch.tensor([3.])]
        total_count = 0
        for value in pieces:
            logits = torch.stack((first * value, -first * value), -1)
            loss, count = objective_sum(logits, first.square())
            (loss / 100).backward(); total_count += count
        first.grad.mul_(100 / total_count)
        all_values = torch.cat(pieces)
        loss, count = objective_sum(torch.stack((second * all_values, -second * all_values), -1), second.square())
        (loss / count).backward()
        self.assertTrue(torch.allclose(first.grad, second.grad, atol=1e-6))

    def test_best_pointer_and_last_two_retention_survive_resume(self):
        import torch
        from marine_ssl.experiments.pretrain import _save_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            output, history, best = Path(folder), [], None
            for step in range(1, 6):
                history, best = _save_checkpoint(output, {"global_step": step, "signature": "same", "best_checkpoint": best,
                                                         "model": {"x": torch.tensor(step)}}, history, is_best=step == 1, keep_last=2)
            self.assertEqual({path.name for path in (output / "checkpoints").iterdir()},
                             {"checkpoint_000000001.pt", "checkpoint_000000004.pt", "checkpoint_000000005.pt"})
            best_state = torch.load(output / "checkpoints" / best, weights_only=False)
            self.assertEqual(best_state["best_checkpoint"], best)
            latest = json.loads((output / "latest.json").read_text())
            self.assertEqual(latest["best_checkpoint"], best)

    def test_small_official_model_backward_checkpoint_smoke(self):
        from marine_ssl.experiments.pretrain import smoke
        with tempfile.TemporaryDirectory() as folder:
            result = smoke(folder)
            self.assertEqual(result["status"], "synthetic_smoke_passed")
            self.assertEqual(result["classes_with_positive"], 201)
            self.assertTrue(result["unmasked_skipped"])

    def test_actual_tar_train_interrupt_resume_matches_uninterrupted(self):
        import hashlib
        import numpy as np
        import sqlite3
        import torch
        from test_codebook import fixture
        from marine_ssl.experiments.codebook import digest, index_info
        from marine_ssl.experiments.pretrain import build_model, make_optimizer, train
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = fixture(root)
            labels_dir = Path(config["labels_dir"]); labels_dir.mkdir()
            database = labels_dir / "labels.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE recordings (tar_path TEXT PRIMARY KEY, members TEXT, labels BLOB, checksum TEXT)")
                for split in ("train", "validation"):
                    row = json.loads(Path(config[f"{split}_index"]).read_text())
                    raw = np.zeros((1, 249), dtype="uint8").tobytes()
                    connection.execute("INSERT INTO recordings VALUES (?,?,?,?)", (row["tar_path"],
                        json.dumps(["audio/00000000000000.wav"]), raw, hashlib.sha256(raw).hexdigest()))
            (labels_dir / "labels.ready.json").write_text(json.dumps({"signature": "synthetic-targets",
                "database_sha256": digest(database), "indexes": {split: index_info(config, split) for split in ("train", "validation")}}))
            config.update(initial_encoder={"kind": "synthetic_test_only"}, seed=31, device="cpu", precision="fp32",
                          epochs=3, warmup_updates=1, microbatch_size=70, accumulation_steps=2,
                          validation_interval_updates=2, checkpoint_interval_updates=2)
            def student(*_):
                model = build_model(tiny=True)
                model.input_normalize_waveform = False
                return model
            optimizer_rates = []
            def recording_optimizer(parameters):
                optimizer = make_optimizer(parameters)
                rates = []; optimizer_rates.append(rates)
                original_step = optimizer.step
                def step(*args, **kwargs):
                    rates.append(optimizer.param_groups[0]["lr"])
                    return original_step(*args, **kwargs)
                optimizer.step = step
                return optimizer
            with patch("marine_ssl.experiments.pretrain.initialize_student", side_effect=student), \
                 patch("marine_ssl.experiments.pretrain.make_optimizer", side_effect=recording_optimizer):
                complete = train({**config, "output_dir": str(root / "complete")})
                pilot = train({**config, "output_dir": str(root / "resumed")}, stop_after_updates=1)
                resumed = train({**config, "output_dir": str(root / "resumed")}, resume="auto")
            self.assertEqual(pilot["status"], "pilot_stopped")
            self.assertGreater(pilot["training_seconds_before_final_checkpoint"], 0)
            self.assertGreaterEqual(pilot["final_checkpoint_seconds"], 0)
            self.assertEqual(pilot["steady_updates"], 0)
            self.assertIsNone(pilot["steady_training_seconds"])
            self.assertEqual(complete["step"], resumed["step"])
            self.assertEqual(optimizer_rates[0][0], 0)
            self.assertEqual(optimizer_rates[0][-1], learning_rate(2, 3, warmup_updates=1))
            self.assertEqual(optimizer_rates[0], optimizer_rates[1] + optimizer_rates[2])
            models = []
            for name in ("complete", "resumed"):
                latest = json.loads((root / name / "latest.json").read_text())
                models.append(torch.load(root / name / "checkpoints" / latest["name"], weights_only=False)["model"])
            self.assertTrue(all(torch.equal(value, models[1][key]) for key, value in models[0].items()))


if __name__ == "__main__":
    unittest.main()
