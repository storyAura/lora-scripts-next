import unittest

from mikazuki.utils.config_import import analyze_train_type, infer_train_type, validate_config_import


class ConfigImportTests(unittest.TestCase):
    def test_standard_anima_lokr_on_fast_page_redirects_to_standard_mode(self):
        config = {
            "model_train_type": "anima-lora",
            "lora_type": "lokr",
            "network_module": "lycoris.kohya",
            "qwen3": "qwen_3_06b_base.safetensors",
            "network_args": ["algo=lokr", "factor=-1"],
        }

        result = validate_config_import("anima-lora-fast", config)

        self.assertEqual(result["result"], "redirect")
        self.assertEqual(result["target_path"], "/lora/sd3.html")

    def test_native_anima_fast_config_is_accepted_on_fast_page(self):
        config = {
            "model_train_type": "anima-lora-fast",
            "lora_type": "lora",
            "method": "lora",
            "methods_subdir": "gui-methods",
            "static_token_count": 4096,
            "compile_mode": "blocks",
            "qwen3": "qwen_3_06b_base.safetensors",
        }

        result = validate_config_import("anima-lora-fast", config)

        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["model_train_type"], "anima-lora-fast")

    def test_native_anima_fast_config_on_standard_page_redirects_to_fast_mode(self):
        config = {
            "model_train_type": "anima-lora-fast",
            "method": "lora",
            "methods_subdir": "gui-methods",
            "static_token_count": 4096,
            "compile_mode": "blocks",
            "qwen3": "qwen_3_06b_base.safetensors",
        }

        result = validate_config_import("sd3-lora", config)

        self.assertEqual(result["result"], "redirect")
        self.assertEqual(result["target_path"], "/lora/anima-fast.html")

    def test_sdxl_on_anima_page_redirects(self):
        config = {
            "model_train_type": "sdxl-lora",
            "pretrained_model_name_or_path": "./sd-models/sdxl/model.safetensors",
            "train_data_dir": "./train/data",
            "max_train_epochs": 10,
        }
        result = validate_config_import("sd3-lora", config)
        self.assertEqual(result["result"], "redirect")
        self.assertEqual(result["target_path"], "/lora/master.html")

    def test_stale_sdxl_type_with_anima_fields_allowed_on_anima_page(self):
        config = {
            "model_train_type": "sdxl-lora",
            "pretrained_model_name_or_path": "anima-base-v1.0.safetensors",
            "qwen3": "qwen_3_06b_base.safetensors",
            "vae": "qwen_image_vae.safetensors",
            "network_module": "lycoris.kohya",
        }
        result = validate_config_import("sd3-lora", config)
        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["model_train_type"], "anima-lora")
        self.assertIn("notice", result)
        self.assertIn("sdxl-lora", result["notice"])
        self.assertIn("anima-lora", result["notice"])
        self.assertTrue(result["detection_reasons"])

    def test_infer_anima_from_model_paths_only(self):
        config = {
            "model_train_type": "sdxl-lora",
            "pretrained_model_name_or_path": "E:/SD-Trainer/sd-models/anima/anima-base-v1.0.safetensors",
            "vae": "E:/SD-Trainer/sd-models/anima/qwen_image_vae.safetensors",
            "qwen3": "E:/SD-Trainer/sd-models/anima/qwen_3_06b_base.safetensors",
        }
        analysis = analyze_train_type(config)
        self.assertEqual(analysis.train_type, "anima-lora")
        self.assertGreaterEqual(len(analysis.reasons), 3)

    def test_redirect_message_includes_detection_reasons(self):
        config = {
            "model_train_type": "sdxl-lora",
            "pretrained_model_name_or_path": "anima-base-v1.0.safetensors",
            "qwen3": "qwen_3_06b_base.safetensors",
            "vae": "qwen_image_vae.safetensors",
        }
        result = validate_config_import("lora-master", config)
        self.assertEqual(result["result"], "redirect")
        self.assertIn("依据", result["message"])
        self.assertIn("qwen3", result["message"])

    def test_missing_train_type_on_anima_page_gets_default(self):
        config = {
            "pretrained_model_name_or_path": "anima-base-v1.0.safetensors",
            "qwen3": "qwen_3_06b_base.safetensors",
            "vae": "qwen_image_vae.safetensors",
        }
        result = validate_config_import("sd3-lora", config)
        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["model_train_type"], "anima-lora")

    def test_legacy_sd3_train_type_on_anima_page(self):
        config = {
            "model_train_type": "sd3-lora",
            "qwen3": "qwen_3_06b_base.safetensors",
            "vae": "qwen_image_vae.safetensors",
        }
        result = validate_config_import("sd3-lora", config)
        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["model_train_type"], "anima-lora")

    def test_sdxl_config_on_master_page_ok(self):
        config = {
            "model_train_type": "sdxl-lora",
            "pretrained_model_name_or_path": "./sd-models/sdxl/model.safetensors",
        }
        result = validate_config_import("lora-master", config)
        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["model_train_type"], "sdxl-lora")

    def test_anima_config_on_master_page_redirects(self):
        config = {
            "model_train_type": "anima-lora",
            "qwen3": "qwen_3_06b_base.safetensors",
            "vae": "qwen_image_vae.safetensors",
        }
        result = validate_config_import("lora-master", config)
        self.assertEqual(result["result"], "redirect")
        self.assertEqual(result["target_path"], "/lora/sd3.html")

    def test_infer_anima_from_network_module(self):
        config = {"network_module": "networks.lora_anima"}
        self.assertEqual(infer_train_type(config), "anima-lora")

    def test_reject_non_object_config(self):
        result = validate_config_import("sd3-lora", "not-a-dict")  # type: ignore[arg-type]
        self.assertEqual(result["result"], "reject")

    def test_reject_sd_scripts_intermediate_toml(self):
        config = {
            "network_module": "networks.lora_anima",
            "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
            "train_data_dir": "./train",
            "max_train_epochs": 2,
        }
        result = validate_config_import("sd3-lora", config)
        self.assertEqual(result["result"], "reject")
        self.assertTrue(any("sd-scripts" in err for err in result["errors"]))

    def test_legacy_preview_fields_add_enable_preview_on_import(self):
        config = {
            "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
            "vae": "./sd-models/anima/qwen_image_vae.safetensors",
            "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
            "network_module": "networks.lora_anima",
            "sample_at_first": True,
            "sample_every_n_epochs": 2,
            "sample_prompts": "./config/autosave/demo-promopt.txt",
        }
        result = validate_config_import("sd3-lora", config)
        self.assertEqual(result["result"], "ok")
        self.assertTrue(result["config"]["enable_preview"])
        self.assertEqual(result["config"]["sample_every_n_epochs"], 2)

    def test_history_row_wrapper_unwraps_before_import(self):
        inner = {
            "model_train_type": "anima-lora",
            "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
            "positive_prompts": "1girl",
            "sample_at_first": True,
        }
        wrapper = {"time": "2026-06-27", "name": "demo", "value": inner}
        result = validate_config_import("sd3-lora", wrapper)
        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["model_train_type"], "anima-lora")
        self.assertNotIn("time", result["config"])
        self.assertNotIn("value", result["config"])
        self.assertTrue(result["config"]["enable_preview"])

    def test_cannot_import_toml_lokr_preview_signals(self):
        import toml

        cfg = toml.loads(
            """
model_train_type = "anima-lora"
lora_type = "lokr"
network_module = "lycoris.kohya"
positive_prompts = "portrait"
network_args = [
  "conv_dim=16",
  "conv_alpha=1",
  "dropout=0",
  "algo=lokr",
  "factor=-1"
]
optimizer_args = ["decouple=True", "weight_deca"]
"""
        )
        result = validate_config_import("sd3-lora", cfg)
        self.assertEqual(result["result"], "ok")
        self.assertTrue(result["config"]["enable_preview"])
        self.assertEqual(result["config"]["lora_type"], "lokr")
        self.assertEqual(result["config"]["conv_dim"], 16)
        self.assertEqual(result["config"]["conv_alpha"], 1)
        self.assertEqual(result["config"]["dropout"], 0)
        self.assertEqual(result["config"]["lycoris_algo"], "lokr")
        self.assertEqual(result["config"]["lokr_factor"], -1)
        self.assertNotIn("weight_deca", result["config"].get("optimizer_args", []))

    def test_lokr627_poisoned_undefined_network_args_sanitized(self):
        config = {
            "model_train_type": "anima-lora",
            "network_module": "lycoris.kohya",
            "network_args": [
                "algo=lokr",
                "conv_dim=undefined",
                "conv_alpha=undefined",
                "dropout=undefined",
                "factor=-1",
            ],
        }
        result = validate_config_import("sd3-lora", config)
        self.assertEqual(result["result"], "ok")
        args = result["config"].get("network_args") or []
        self.assertIn("algo=lokr", args)
        self.assertIn("factor=-1", args)
        self.assertFalse(any("undefined" in item for item in args))
        self.assertEqual(result["config"]["lycoris_algo"], "lokr")
        self.assertEqual(result["config"]["lokr_factor"], -1)

    def test_target_res_string_hydrates_to_checkbox_array(self):
        result = validate_config_import(
            "sd3-lora",
            {
                "model_train_type": "anima-lora",
                "qwen3": "qwen_3_06b_base.safetensors",
                "multires_per_image": True,
                "target_res": "512,1024，768",
            },
        )
        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["target_res"], ["512", "1024", "768"])

    def test_target_res_already_array_keeps_allowed_tiers(self):
        result = validate_config_import(
            "sd3-lora",
            {
                "model_train_type": "anima-lora",
                "qwen3": "qwen_3_06b_base.safetensors",
                "target_res": [512, "1024", "999", "1024"],
            },
        )
        self.assertEqual(result["result"], "ok")
        self.assertEqual(result["config"]["target_res"], ["512", "1024"])


class AnimaLoraTypeBranchConstTests(unittest.TestCase):
    """History snapshots lack hidden union consts (network_module / lycoris_algo).

    The frontend fullReplace merge backfills them from the union's FIRST branch
    defaults, which poisons every non-"lora" branch and blanks the form + TOML
    preview. The backend must derive them from lora_type so the merge overrides
    the poisoned defaults.
    """

    EXPECTED = {
        "lora": ("networks.lora_anima", None),
        "rslora": ("lycoris.kohya", "lora"),
        "lora_plus": ("networks.lora_anima", None),
        "dora": ("lycoris.kohya", "lora"),
        "lora_fa": ("networks.lora_fa_anima", None),
        "vera": ("networks.vera_anima", None),
        "delora": ("networks.delora_anima", None),
        "waveft": ("networks.waveft_anima", None),
        "deft": ("networks.deft_anima", None),
        "moslora": ("networks.moslora_anima", None),
        "tlora": ("networks.tlora_anima", None),
        "loha": ("networks.loha", None),
        "lokr": ("lycoris.kohya", "lokr"),
        "cdka": ("networks.cdka_anima", None),
        "bokr": ("lycoris.kohya", "bokr"),
        "bora": ("lycoris.kohya", "bora"),
        "gsokr": ("lycoris.kohya", "gsokr"),
        "glora_boft": ("lycoris.kohya", "glora_boft"),
    }

    def test_every_lora_type_snapshot_gets_branch_consts(self):
        for lora_type, (module, algo) in self.EXPECTED.items():
            with self.subTest(lora_type=lora_type):
                snapshot = {
                    "model_train_type": "anima-lora",
                    "lora_type": lora_type,
                    "network_dim": 32,
                }
                result = validate_config_import("sd3-lora", snapshot)
                self.assertEqual(result["result"], "ok")
                self.assertEqual(result["config"]["network_module"], module)
                if algo is not None:
                    self.assertEqual(result["config"]["lycoris_algo"], algo)

    def test_stale_lycoris_algo_from_other_branch_is_corrected(self):
        snapshot = {
            "model_train_type": "anima-lora",
            "lora_type": "bokr",
            "lycoris_algo": "glokr",
            "network_module": "lycoris.kohya",
        }
        result = validate_config_import("sd3-lora", snapshot)
        self.assertEqual(result["config"]["lycoris_algo"], "bokr")

    def test_stale_network_module_from_other_branch_is_corrected(self):
        snapshot = {
            "model_train_type": "anima-lora",
            "lora_type": "tlora",
            "network_module": "lycoris.kohya",
        }
        result = validate_config_import("sd3-lora", snapshot)
        self.assertEqual(result["config"]["network_module"], "networks.tlora_anima")

    def test_lora_type_recovered_from_unambiguous_network_module(self):
        for module, expected in (
            ("networks.tlora_anima", "tlora"),
            ("networks.delora_anima", "delora"),
            ("networks.waveft_anima", "waveft"),
            ("networks.deft_anima", "deft"),
            ("networks.moslora_anima", "moslora"),
            ("networks.loha", "loha"),
        ):
            with self.subTest(module=module):
                config = {
                    "model_train_type": "anima-lora",
                    "network_module": module,
                }
                result = validate_config_import("sd3-lora", config)
                self.assertEqual(result["config"]["lora_type"], expected)

    def test_custom_algorithm_network_args_are_hydrated_for_the_form(self):
        cases = (
            (
                "networks.delora_anima",
                ["delora_lambda=12.5"],
                {"delora_lambda": 12.5},
            ),
            (
                "networks.waveft_anima",
                [
                    "waveft_n_frequency=128",
                    "waveft_use_idwt=False",
                    "waveft_wavelet_family=db1",
                ],
                {
                    "waveft_n_frequency": 128,
                    "waveft_use_idwt": False,
                    "waveft_wavelet_family": "db1",
                },
            ),
            (
                "networks.deft_anima",
                [
                    "deft_decomposition_method=qr",
                    "deft_alpha=16",
                    "deft_init_weights=False",
                ],
                {
                    "deft_decomposition_method": "qr",
                    "deft_alpha": 16,
                    "deft_init_weights": False,
                },
            ),
            (
                "networks.moslora_anima",
                ["moslora_mixer_init=orthogonal"],
                {"moslora_mixer_init": "orthogonal"},
            ),
        )
        for network_module, network_args, expected in cases:
            with self.subTest(network_module=network_module):
                result = validate_config_import(
                    "sd3-lora",
                    {
                        "model_train_type": "anima-lora",
                        "network_module": network_module,
                        "network_args": network_args,
                    },
                )
                for key, value in expected.items():
                    self.assertEqual(result["config"][key], value)

    def test_non_anima_page_is_untouched(self):
        config = {"model_train_type": "flux-lora", "lora_type": "lokr"}
        result = validate_config_import("flux-lora", config)
        self.assertEqual(result["result"], "ok")
        self.assertNotIn("network_module", result["config"])


class QueueEditHandoverRoundTripTests(unittest.TestCase):
    """The queue 「编辑」 flow hands the stored /api/run body to validate-import.

    That body is parseParams output: string LRs parseFloat'ed to numbers and
    every LyCORIS branch field folded into network_args then deleted
    (needDeleteParams). The backend must hydrate the UI fields back so the
    form's fullReplace merge restores the entry instead of blanking it.
    """

    def test_flat_posted_lokr_config_round_trips_for_the_form(self):
        posted = {
            "model_train_type": "anima-lora",
            "lora_type": "lokr",
            "pretrained_model_name_or_path": "./sd-models/anima-base.safetensors",
            "train_data_dir": "./train/aki",
            "learning_rate": 0.0001,
            "unet_lr": 0.0001,
            "text_encoder_lr": 1e-05,
            "max_train_epochs": 24,
            "train_batch_size": 4,
            "gradient_checkpointing": True,
            "network_dim": 10000,
            "network_alpha": 1,
            "network_module": "lycoris.kohya",
            "network_args": [
                "conv_dim=100000",
                "conv_alpha=1",
                "dropout=0",
                "algo=lokr",
                "factor=8",
                "full_matrix=True",
            ],
        }
        result = validate_config_import("sd3-lora", posted)
        self.assertEqual(result["result"], "ok")
        config = result["config"]
        # branch consts stamped so the lora_type union keeps matching
        self.assertEqual(config["network_module"], "lycoris.kohya")
        self.assertEqual(config["lycoris_algo"], "lokr")
        # deleted branch fields hydrated back from network_args
        self.assertEqual(config["lokr_factor"], 8)
        self.assertEqual(config["conv_dim"], 100000)
        self.assertEqual(config["dropout"], 0)
        self.assertIs(config["full_matrix"], True)
        # user values pass through untouched (frontend re-formats LR strings)
        self.assertEqual(config["learning_rate"], 0.0001)
        self.assertEqual(config["max_train_epochs"], 24)
        self.assertEqual(config["model_train_type"], "anima-lora")


if __name__ == "__main__":
    unittest.main()
