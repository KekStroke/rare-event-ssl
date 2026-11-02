import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


@unittest.skipUnless(HAS_STACK, "PyTorch/TorchAudio/NumPy required")
class PretrainStackTests(unittest.TestCase):
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
