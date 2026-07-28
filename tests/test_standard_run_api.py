from __future__ import annotations

import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from starlette.requests import Request

from mikazuki.app import api


def make_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/api/run", "headers": []}, receive)


class StandardRunApiTests(unittest.TestCase):
    def test_anima_run_rejects_sageattention_before_path_validation(self):
        response = asyncio.run(api.create_toml_file(make_request({
            "model_train_type": "anima-lora",
            "attn_mode": "sageattn",
        })))

        self.assertEqual(response.status, "fail")
        self.assertEqual(response.data["field"], "attn_mode")
        self.assertIn("does not support training backward", response.message)

    def test_anima_run_accepts_vera_and_continues_to_path_validation(self):
        response = asyncio.run(api.create_toml_file(make_request({
            "model_train_type": "anima-lora",
            "lora_type": "vera",
        })))

        self.assertEqual(response.status, "fail")
        self.assertEqual(response.data["field"], "train_data_dir")

    def test_anima_run_accepts_pissa_and_continues_to_path_validation(self):
        response = asyncio.run(api.create_toml_file(make_request({
            "model_train_type": "anima-lora",
            "lora_type": "lora",
            "pissa_init": True,
        })))

        self.assertEqual(response.status, "fail")
        self.assertEqual(response.data["field"], "train_data_dir")

    def test_run_rejects_lora_plus_with_prodigy_before_path_validation(self):
        response = asyncio.run(api.create_toml_file(make_request({
            "model_train_type": "anima-lora",
            "lora_type": "lora_plus",
            "optimizer_type": "Prodigy",
        })))

        self.assertEqual(response.status, "fail")
        self.assertEqual(response.data["field"], "optimizer_type")
        self.assertIn("LoRA+", response.message)

    def test_run_converts_unusable_requested_attention_to_structured_failure(self):
        probe = types.SimpleNamespace(usable=False, reason="backward probe failed")
        with mock.patch.object(api, "probe_training_attention_backend", return_value=probe):
            response = asyncio.run(api.create_toml_file(make_request({
                "model_train_type": "anima-lora",
                "attn_mode": "xformers",
                "train_data_dir": "E:/OpenSourceTeamWork/not-used",
                "pretrained_model_name_or_path": "not-used-model.safetensors",
            })))

        self.assertEqual(response.status, "fail")
        self.assertEqual(response.data["field"], "attn_mode")
        self.assertEqual(response.data["value"], "xformers")
        self.assertIn("probe failed", response.message)

    def test_run_rejects_unknown_standard_train_type_without_500(self):
        response = asyncio.run(api.create_toml_file(make_request({"model_train_type": "unknown-lora"})))

        self.assertEqual(response.status, "fail")
        self.assertIn("不支持的训练类型", response.message)
        self.assertEqual(response.data["model_train_type"], "unknown-lora")

    def test_run_rejects_missing_train_data_dir_without_connect_error(self):
        response = asyncio.run(api.create_toml_file(make_request({
            "model_train_type": "sd-lora",
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        })))

        self.assertEqual(response.status, "fail")
        self.assertEqual(response.data["field"], "train_data_dir")
        self.assertIn("训练数据集路径", response.message)

    def test_run_rejects_missing_model_path_without_connect_error(self):
        response = asyncio.run(api.create_toml_file(make_request({
            "model_train_type": "sd-lora",
            "train_data_dir": "E:/OpenSourceTeamWork/not-used",
        })))

        self.assertEqual(response.status, "fail")
        self.assertEqual(response.data["field"], "pretrained_model_name_or_path")
        self.assertIn("底模路径", response.message)

    def test_run_starts_standard_lora_with_task_metadata_response(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "dataset"
            (data_dir / "1_class").mkdir(parents=True)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

            payload = {
                "model_train_type": "sd-lora",
                "train_data_dir": str(data_dir),
                "pretrained_model_name_or_path": str(model_dir),
                "output_dir": str(root / "output"),
                "output_name": "unit-standard-lora",
                "enable_preview": False,
            }
            fake_response = api.APIResponseSuccess(
                message="Training started",
                data={
                    "task_id": "task-standard",
                    "train_log_url": "http://127.0.0.1:28000/train-log?task_id=task-standard",
                    "metadata": {"backend": "standard", "trainer_file": "./scripts/stable/train_network.py"},
                },
            )

            with mock.patch.object(api.os, "getcwd", return_value=str(root)), \
                    mock.patch.object(api.process, "run_train", return_value=fake_response) as run_train:
                response = asyncio.run(api.create_toml_file(make_request(payload)))

            self.assertEqual(response.status, "success")
            self.assertEqual(response.data["task_id"], "task-standard")
            self.assertIn("train_log_url", response.data)
            run_train.assert_called_once()
            toml_path, trainer_file, _gpu_ids, cpu_threads = run_train.call_args.args
            self.assertTrue(Path(toml_path).is_file())
            self.assertEqual(trainer_file, "./scripts/stable/train_network.py")
            self.assertEqual(cpu_threads, 2)

    def test_run_routes_sdxl_lora_to_vendor_trainer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "dataset"
            image_dir = data_dir / "1_class"
            image_dir.mkdir(parents=True)
            (image_dir / "sample.png").write_bytes(b"not-used-by-mocked-runner")
            model_path = root / "model.safetensors"
            model_path.write_bytes(b"not-used-by-mocked-runner")

            payload = {
                "model_train_type": "sdxl-lora",
                "train_data_dir": str(data_dir),
                "pretrained_model_name_or_path": str(model_path),
                "output_dir": str(root / "output"),
                "output_name": "unit-sdxl-lora",
                "enable_preview": False,
            }
            fake_response = api.APIResponseSuccess(
                message="Training started",
                data={
                    "task_id": "task-sdxl",
                    "metadata": {"backend": "standard", "trainer_file": "./vendor/sd-scripts/sdxl_train_network.py"},
                },
            )

            with mock.patch.object(api.os, "getcwd", return_value=str(root)), \
                    mock.patch.object(api.process, "run_train", return_value=fake_response) as run_train:
                response = asyncio.run(api.create_toml_file(make_request(payload)))

            self.assertEqual(response.status, "success")
            run_train.assert_called_once()
            _toml_path, trainer_file, _gpu_ids, cpu_threads = run_train.call_args.args
            self.assertEqual(trainer_file, "./vendor/sd-scripts/sdxl_train_network.py")
            self.assertEqual(cpu_threads, 2)


if __name__ == "__main__":
    unittest.main()
