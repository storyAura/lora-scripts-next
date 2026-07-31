import unittest

from mikazuki.anima_backend.adapter import adapt_anima_config


class AnimaBackendAdapterTests(unittest.TestCase):
    def test_rslora_type_maps_to_lycoris_with_rank_stabilized_scaling(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "rslora",
                "network_dim": 16,
                "network_alpha": 16,
            }
        )

        self.assertEqual(adapted["network_module"], "lycoris.kohya")
        self.assertIn("algo=lora", adapted["network_args"])
        self.assertIn("rs_lora=True", adapted["network_args"])
        self.assertEqual(warnings, [])

    def test_dora_type_maps_to_lycoris_weight_decomposition(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "dora",
                "network_dim": 16,
                "network_alpha": 16,
            }
        )

        self.assertEqual(adapted["network_module"], "lycoris.kohya")
        self.assertIn("algo=lora", adapted["network_args"])
        self.assertIn("dora_wd=True", adapted["network_args"])
        self.assertEqual(warnings, [])

    def test_lora_plus_type_injects_strict_lr_ratio(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "lora_plus",
                "loraplus_lr_ratio": 16,
            }
        )

        self.assertEqual(adapted["network_module"], "networks.lora_anima")
        self.assertIn("loraplus_lr_ratio=16", adapted["network_args"])
        self.assertNotIn("loraplus_lr_ratio", adapted)
        self.assertEqual(warnings, [])

    def test_lorafa_type_selects_real_network_and_required_optimizer(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "lora_fa",
                "optimizer_type": "AdamW8bit",
            }
        )

        self.assertEqual(adapted["network_module"], "networks.lora_fa_anima")
        self.assertEqual(adapted["optimizer_type"], "LoRAFAAdamW")
        self.assertTrue(any("LoRAFAAdamW" in warning for warning in warnings))

    def test_vera_type_selects_real_network_and_projection_contract(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "vera",
                "vera_projection_seed": 42,
                "vera_save_projection": True,
                "vera_d_initial": 0.1,
            }
        )

        self.assertEqual(adapted["network_module"], "networks.vera_anima")
        self.assertIn("vera_projection_seed=42", adapted["network_args"])
        self.assertIn("vera_save_projection=True", adapted["network_args"])
        self.assertIn("vera_d_initial=0.1", adapted["network_args"])
        self.assertEqual(warnings, [])

    def test_new_algorithm_fields_select_real_networks_and_are_forwarded(self):
        cases = (
            (
                {
                    "lora_type": "delora",
                    "delora_lambda": 12.5,
                },
                "networks.delora_anima",
                {"delora_lambda=12.5"},
            ),
            (
                {
                    "lora_type": "waveft",
                    "waveft_n_frequency": 128,
                    "waveft_scaling": 20,
                    "waveft_random_loc_seed": 9,
                    "waveft_use_idwt": False,
                    "waveft_wavelet_family": "db1",
                },
                "networks.waveft_anima",
                {
                    "waveft_n_frequency=128",
                    "waveft_scaling=20",
                    "waveft_random_loc_seed=9",
                    "waveft_use_idwt=False",
                    "waveft_wavelet_family=db1",
                },
            ),
            (
                {
                    "lora_type": "deft",
                    "deft_decomposition_method": "qr",
                    "deft_alpha": 16,
                    "deft_init_scale": 1.5,
                    "deft_init_weights": False,
                },
                "networks.deft_anima",
                {
                    "deft_decomposition_method=qr",
                    "deft_alpha=16",
                    "deft_init_scale=1.5",
                    "deft_init_weights=False",
                },
            ),
            (
                {
                    "lora_type": "moslora",
                    "moslora_mixer_init": "orthogonal",
                },
                "networks.moslora_anima",
                {"moslora_mixer_init=orthogonal"},
            ),
        )

        for config, expected_module, expected_args in cases:
            with self.subTest(lora_type=config["lora_type"]):
                adapted, warnings = adapt_anima_config(config)
                self.assertEqual(adapted["network_module"], expected_module)
                self.assertTrue(
                    expected_args.issubset(set(adapted["network_args"]))
                )
                self.assertEqual(warnings, [])

    def test_false_vera_projection_flag_is_not_silently_dropped(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "vera",
                "vera_save_projection": False,
            }
        )

        self.assertIn("vera_save_projection=False", adapted["network_args"])
        self.assertEqual(warnings, [])

    def test_pissa_fields_are_forwarded_to_the_lora_network(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "lora",
                "pissa_init": True,
                "pissa_method": "rsvd",
                "pissa_niter": 3,
                "pissa_oversample": 6,
                "pissa_apply_conv2d": True,
                "pissa_export_mode": "LoRA无损兼容导出",
            }
        )

        self.assertEqual(adapted["network_module"], "networks.lora_anima")
        self.assertIn("pissa_init=True", adapted["network_args"])
        self.assertIn("pissa_method=rsvd", adapted["network_args"])
        self.assertIn("pissa_niter=3", adapted["network_args"])
        self.assertIn("pissa_oversample=6", adapted["network_args"])
        self.assertIn("pissa_apply_conv2d=True", adapted["network_args"])
        self.assertIn(
            "pissa_export_mode=LoRA无损兼容导出",
            adapted["network_args"],
        )
        self.assertEqual(warnings, [])

    def test_adapter_keeps_supported_anima_fields(self):
        config = {
            "model_train_type": "anima-lora",
            "pretrained_model_name_or_path": "./sd-models/anima/anima-preview3-base.safetensors",
            "vae": "./sd-models/anima/qwen_image_vae.safetensors",
            "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
            "network_module": "networks.lora_anima",
            "network_dim": 16,
            "network_alpha": 16,
            "enable_preview": True,
            "sample_width": 1024,
            "sample_height": 1024,
        }

        adapted, warnings = adapt_anima_config(config)

        self.assertTrue(adapted["pretrained_model_name_or_path"].endswith("anima-preview3-base.safetensors"))
        self.assertTrue(adapted["vae"].endswith("qwen_image_vae.safetensors"))
        self.assertTrue(adapted["qwen3"].endswith("qwen_3_06b_base.safetensors"))
        self.assertEqual(adapted["network_module"], "networks.lora_anima")
        self.assertEqual(adapted["network_dim"], 16)
        self.assertNotIn("model_train_type", adapted)
        self.assertNotIn("enable_preview", adapted)
        self.assertEqual(warnings, [])

    def test_adapter_warns_for_unsupported_debug_fields(self):
        adapted, warnings = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "anima_debug_mode": True,
                "anima_rope_mismatch_mode": "resample",
            }
        )

        self.assertEqual(adapted["pretrained_model_name_or_path"], "model.safetensors")
        self.assertNotIn("anima_debug_mode", adapted)
        self.assertNotIn("anima_rope_mismatch_mode", adapted)
        self.assertIn("Unsupported Anima field ignored: anima_debug_mode", warnings)
        self.assertIn("Unsupported Anima field ignored: anima_rope_mismatch_mode", warnings)

    def test_network_args_custom_becomes_network_args(self):
        adapted, warnings = adapt_anima_config(
            {
                "network_args_custom": ["train_llm_adapter=True", "verbose=True"],
            }
        )

        self.assertEqual(adapted["network_args"], ["train_llm_adapter=True", "verbose=True"])
        self.assertNotIn("network_args_custom", adapted)
        self.assertEqual(warnings, [])

    def test_adapter_warns_when_unknown_field_is_passed_through(self):
        adapted, warnings = adapt_anima_config({"future_sd_scripts_option": "enabled"})

        self.assertEqual(adapted["future_sd_scripts_option"], "enabled")
        self.assertIn("Unknown field passed through to sd-scripts: future_sd_scripts_option", warnings)

    def test_tlora_fields_injected_into_network_args(self):
        config = {
            "network_module": "networks.tlora_anima",
            "network_dim": 16,
            "network_alpha": 16,
            "tlora_min_rank": 2,
            "tlora_rank_schedule": "linear",
            "tlora_orthogonal_init": True,
        }
        adapted, warnings = adapt_anima_config(config)

        self.assertIn("network_args", adapted)
        self.assertIn("tlora_min_rank=2", adapted["network_args"])
        self.assertIn("tlora_rank_schedule=linear", adapted["network_args"])
        self.assertIn("tlora_orthogonal_init=True", adapted["network_args"])
        self.assertNotIn("tlora_min_rank", adapted)
        self.assertNotIn("tlora_rank_schedule", adapted)
        self.assertNotIn("tlora_orthogonal_init", adapted)
        self.assertEqual(warnings, [])

    def test_tlora_fields_merge_with_existing_network_args(self):
        config = {
            "network_module": "networks.tlora_anima",
            "network_args": ["verbose=True"],
            "tlora_min_rank": 4,
        }
        adapted, warnings = adapt_anima_config(config)

        self.assertIn("verbose=True", adapted["network_args"])
        self.assertIn("tlora_min_rank=4", adapted["network_args"])

    def test_non_tlora_module_ignores_tlora_fields(self):
        config = {
            "network_module": "networks.lora_anima",
            "tlora_min_rank": 2,
        }
        adapted, warnings = adapt_anima_config(config)

        self.assertNotIn("tlora_min_rank", adapted)
        self.assertEqual(warnings, [])

    def test_lora_type_is_ui_only(self):
        config = {
            "lora_type": "tlora",
            "network_module": "networks.tlora_anima",
        }
        adapted, warnings = adapt_anima_config(config)

        self.assertNotIn("lora_type", adapted)

    def test_lycoris_fields_injected_into_network_args(self):
        config = {
            "network_module": "lycoris.kohya",
            "network_dim": 16,
            "network_alpha": 16,
            "lycoris_algo": "lokr",
            "lokr_factor": -1,
            "use_cp": True,
            "decompose_both": True,
            "use_scalar": False,
            "dora_wd": True,
            "full_matrix": True,
            "bypass_mode": False,
            "dropout": 0.1,
            "rank_dropout": 0.05,
            "module_dropout": 0.0,
        }
        adapted, warnings = adapt_anima_config(config)

        na = adapted["network_args"]
        self.assertIn("algo=lokr", na)
        self.assertIn("factor=-1", na)
        self.assertIn("use_cp=True", na)
        self.assertIn("decompose_both=True", na)
        self.assertIn("dora_wd=True", na)
        self.assertIn("full_matrix=True", na)
        self.assertIn("dropout=0.1", na)
        self.assertIn("rank_dropout=0.05", na)
        # False values should be omitted (use LyCORIS defaults)
        self.assertNotIn("use_scalar=False", na)
        self.assertNotIn("bypass_mode=False", na)
        # These should NOT appear as top-level keys
        self.assertNotIn("lycoris_algo", adapted)
        self.assertNotIn("lokr_factor", adapted)
        self.assertNotIn("use_cp", adapted)
        self.assertNotIn("full_matrix", adapted)
        self.assertTrue(any("full_matrix=true" in warning for warning in warnings))

    def test_lycoris_preset_and_fields_coexist(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "lokr_factor": 16,
            "full_matrix": True,
            "network_args": ["verbose=True"],
        }
        adapted, warnings = adapt_anima_config(config)

        na = adapted["network_args"]
        self.assertIn("verbose=True", na)
        self.assertTrue(any(item.startswith("preset=") for item in na))
        self.assertIn("algo=lokr", na)
        self.assertIn("factor=16", na)
        self.assertIn("full_matrix=True", na)

    def test_non_lycoris_module_ignores_lycoris_fields(self):
        config = {
            "network_module": "networks.lora_anima",
            "use_cp": True,
            "lokr_factor": 8,
            "full_matrix": True,
        }
        adapted, warnings = adapt_anima_config(config)

        self.assertNotIn("use_cp", adapted)
        self.assertNotIn("lokr_factor", adapted)
        self.assertNotIn("full_matrix", adapted)
        self.assertEqual(warnings, [])

    def test_lycoris_zero_numeric_values_passed_through(self):
        """Numeric 0 is a valid value (e.g. dropout=0) and should be included."""
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "dropout": 0,
            "module_dropout": 0.0,
        }
        adapted, warnings = adapt_anima_config(config)

        na = adapted["network_args"]
        self.assertIn("dropout=0", na)
        self.assertIn("module_dropout=0.0", na)

    def test_lokr_train_norm_is_disabled_with_warning(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "train_norm": True,
        }

        adapted, warnings = adapt_anima_config(config)

        self.assertIn("network_args", adapted)
        self.assertIn("algo=lokr", adapted["network_args"])
        self.assertNotIn("train_norm=True", adapted["network_args"])
        self.assertTrue(any("train_norm" in warning and "LoKr" in warning for warning in warnings))

    def test_lokr_bf16_warns_and_keeps_weight_decomposition_args(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "mixed_precision": "bf16",
            "full_bf16": True,
            "full_matrix": True,
            "dora_wd": True,
            "network_args": ["factor=-1", "weight_decomposition=True"],
        }

        adapted, warnings = adapt_anima_config(config)

        self.assertIn("network_args", adapted)
        self.assertIn("algo=lokr", adapted["network_args"])
        self.assertIn("factor=-1", adapted["network_args"])
        self.assertIn("full_matrix=True", adapted["network_args"])
        self.assertIn("dora_wd=True", adapted["network_args"])
        self.assertIn("weight_decomposition=True", adapted["network_args"])
        self.assertTrue(adapted["full_bf16"])
        self.assertTrue(
            any("DoRA/weight_decomposition" in warning and "keeps your" in warning for warning in warnings)
        )
        self.assertTrue(any("full_matrix=true" in warning for warning in warnings))

    def test_lokr_full_matrix_autofills_scale_weight_norms_guardrail(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "mixed_precision": "bf16",
            "full_bf16": True,
            "full_matrix": True,
        }

        adapted, warnings = adapt_anima_config(config)

        self.assertIn("full_matrix=True", adapted["network_args"])
        self.assertTrue(adapted["full_bf16"])
        self.assertEqual(adapted["scale_weight_norms"], 1.0)
        self.assertTrue(any("scale_weight_norms=1.0" in warning for warning in warnings))

    def test_lokr_full_matrix_respects_explicit_scale_weight_norms(self):
        for explicit in (0, 2.5):
            with self.subTest(explicit=explicit):
                config = {
                    "network_module": "lycoris.kohya",
                    "lycoris_algo": "lokr",
                    "full_matrix": True,
                    "scale_weight_norms": explicit,
                }

                adapted, warnings = adapt_anima_config(config)

                self.assertEqual(adapted["scale_weight_norms"], explicit)
                self.assertFalse(any("auto-enabled" in warning for warning in warnings))
                self.assertTrue(any("full_matrix=true" in warning for warning in warnings))

    def test_lokr_fp16_keeps_weight_decomposition_args(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "mixed_precision": "fp16",
            "dora_wd": True,
            "network_args": ["weight_decomposition=True"],
        }

        adapted, warnings = adapt_anima_config(config)

        self.assertIn("dora_wd=True", adapted["network_args"])
        self.assertIn("weight_decomposition=True", adapted["network_args"])

    def test_lycoris_non_lokr_keeps_train_norm(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "locon",
            "train_norm": True,
        }

        adapted, warnings = adapt_anima_config(config)

        self.assertIn("algo=locon", adapted["network_args"])
        self.assertIn("train_norm=True", adapted["network_args"])
        self.assertEqual(warnings, [])

    def test_learning_rate_fills_missing_component_lrs(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "network_train_unet_only": True,
            "learning_rate": "1",
            "unet_lr": "",
            "text_encoder_lr": "",
        }
        adapted, warnings = adapt_anima_config(config)

        self.assertEqual(adapted["learning_rate"], "1")
        self.assertEqual(adapted["unet_lr"], "1")
        self.assertEqual(adapted["text_encoder_lr"], "1")
        self.assertEqual(warnings, [])

    def test_learning_rate_does_not_override_component_lrs(self):
        config = {
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "learning_rate": "1",
            "unet_lr": "5e-5",
            "text_encoder_lr": "1e-5",
        }
        adapted, warnings = adapt_anima_config(config)

        self.assertEqual(adapted["unet_lr"], "5e-5")
        self.assertEqual(adapted["text_encoder_lr"], "1e-5")
        self.assertEqual(warnings, [])

    @staticmethod
    def _arg_value(network_args, key):
        found = None
        for item in network_args:
            k, v = item.split("=", 1)
            if k == key:
                found = v
        return found

    def test_stale_glokr_fields_dropped_after_removal(self):
        # GLoKR was removed from the GUI (2026-07-29). Old autosaves still carry
        # its branch fields; every one of them must be dropped silently instead
        # of leaking into network_args or "unknown field" warnings.
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "lokr",
                "network_module": "lycoris.kohya",
                "lycoris_algo": "lokr",
                "kron_rank": 2,
                "use_bora": True,
                "bora_iters": 2,
                "train_gates": True,
                "init_mode": "nkp",
                "use_g_out": True,
                "g_norm_mode": "frobenius",
                "lokr_factor": -1,
            }
        )
        args = adapted["network_args"]

        self.assertEqual(self._arg_value(args, "algo"), "lokr")
        self.assertEqual(self._arg_value(args, "factor"), "-1")
        stale_fields = (
            "kron_rank", "use_bora", "bora_iters",
            "train_gates", "init_mode", "use_g_out", "g_norm_mode",
        )
        for stale in stale_fields:
            self.assertIsNone(self._arg_value(args, stale))
            self.assertNotIn(stale, adapted)
        self.assertEqual(
            [w for w in warnings if any(field in w for field in stale_fields)], []
        )

    def test_gsokr_and_boft_ui_fields_become_network_args(self):
        adapted, _ = adapt_anima_config(
            {
                "network_module": "lycoris.kohya",
                "lycoris_algo": "gsokr",
                "use_sora": True,
                "sora_r": 4,
                "sora_epsilon": 0.00001,
            }
        )
        args = adapted["network_args"]
        self.assertEqual(self._arg_value(args, "use_sora"), "True")
        self.assertEqual(self._arg_value(args, "sora_r"), "4")
        self.assertEqual(float(self._arg_value(args, "sora_epsilon")), 0.00001)

        adapted, _ = adapt_anima_config(
            {
                "network_module": "lycoris.kohya",
                "lycoris_algo": "glora_boft",
                "boft_constraint": 0,
                "boft_rescaled": False,
            }
        )
        args = adapted["network_args"]
        self.assertEqual(self._arg_value(args, "constraint"), "0")
        self.assertIsNone(self._arg_value(args, "rescaled"))
        self.assertIsNone(self._arg_value(args, "boft_constraint"))
        self.assertNotIn("boft_constraint", adapted)
        self.assertNotIn("boft_rescaled", adapted)

    def test_custom_network_args_passthrough_survives_removed_ui_field(self):
        # The stale top-level kron_rank is dropped (field removed from UI), but
        # an explicit custom network_args entry is a power-user channel and the
        # vendored lycoris still accepts it — it must pass through untouched.
        adapted, _ = adapt_anima_config(
            {
                "network_module": "lycoris.kohya",
                "lycoris_algo": "glokr",
                "kron_rank": 2,
                "network_args_custom": ["kron_rank=4", "wd_on_output=False"],
            }
        )
        args = adapted["network_args"]
        kron_values = [a.split("=", 1)[1] for a in args if a.startswith("kron_rank=")]
        self.assertEqual(kron_values, ["4"])
        self.assertEqual(self._arg_value(args, "wd_on_output"), "False")

    def test_finetune_strips_extension_algo_fields(self):
        adapted, _ = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "kron_rank": 2,
                "use_sora": True,
                "boft_constraint": 0,
            },
            finetune=True,
        )
        self.assertNotIn("kron_rank", adapted)
        self.assertNotIn("use_sora", adapted)
        self.assertNotIn("boft_constraint", adapted)
        self.assertNotIn("network_args", adapted)


class LoraTypeOverrideTests(unittest.TestCase):
    """Stale network_module / lycoris_algo left over from another lora_type branch
    (the schema keeps them tolerant so the frontend union survives switching)
    must never leak into training — the adapter derives both from lora_type."""

    @staticmethod
    def _arg_value(network_args, key):
        for item in network_args:
            k, _, v = str(item).partition("=")
            if k.strip() == key:
                return v.strip()
        return None

    def test_stale_lycoris_module_corrected_for_tlora(self):
        adapted, warnings = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "lora_type": "tlora",
                "network_module": "lycoris.kohya",
                "lycoris_algo": "glokr",
                "tlora_min_rank": 4,
            }
        )
        self.assertEqual(adapted["network_module"], "networks.tlora_anima")
        self.assertEqual(self._arg_value(adapted.get("network_args", []), "tlora_min_rank"), "4")
        self.assertNotIn("algo=glokr", adapted.get("network_args", []))
        self.assertTrue(any("networks.tlora_anima" in w for w in warnings))

    def test_stale_algo_corrected_when_switching_lycoris_types(self):
        adapted, warnings = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "lora_type": "bokr",
                "network_module": "lycoris.kohya",
                "lycoris_algo": "glokr",
            }
        )
        self.assertEqual(adapted["network_module"], "lycoris.kohya")
        self.assertEqual(self._arg_value(adapted["network_args"], "algo"), "bokr")
        self.assertTrue(any("bokr" in w for w in warnings))

    def test_removed_glokr_fails_instead_of_training_something_else(self):
        from mikazuki.training_validation import TrainingConfigurationError

        with self.assertRaises(TrainingConfigurationError):
            adapt_anima_config(
                {
                    "pretrained_model_name_or_path": "model.safetensors",
                    "lora_type": "glokr",
                    "network_module": "lycoris.kohya",
                    "lycoris_algo": "glokr",
                }
            )

    def test_removed_tglokr_fails_instead_of_training_plain_glokr(self):
        from mikazuki.training_validation import TrainingConfigurationError

        with self.assertRaises(TrainingConfigurationError):
            adapt_anima_config(
                {
                    "pretrained_model_name_or_path": "model.safetensors",
                    "lora_type": "tglokr",
                    "network_module": "lycoris.kohya",
                    "lycoris_algo": "glokr",
                }
            )

    def test_stale_time_gate_fields_are_dropped_silently(self):
        adapted, warnings = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "lora_type": "lokr",
                "network_module": "lycoris.kohya",
                "lycoris_algo": "lokr",
                "train_time_gates": True,
                "time_gate_dim": 4,
            }
        )
        args = adapted["network_args"]
        self.assertIsNone(self._arg_value(args, "train_time_gates"))
        self.assertIsNone(self._arg_value(args, "time_gate_dim"))
        self.assertNotIn("train_time_gates", adapted)
        self.assertEqual([w for w in warnings if "time_gate" in w], [])

    def test_matching_values_produce_no_warnings(self):
        _, warnings = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "lora_type": "lokr",
                "network_module": "lycoris.kohya",
                "lycoris_algo": "lokr",
            }
        )
        self.assertEqual([w for w in warnings if "不符" in w], [])

    def test_missing_lora_type_keeps_form_values(self):
        adapted, _ = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "network_module": "lycoris.kohya",
                "lycoris_algo": "lokr",
            }
        )
        self.assertEqual(adapted["network_module"], "lycoris.kohya")
        self.assertEqual(self._arg_value(adapted["network_args"], "algo"), "lokr")


if __name__ == "__main__":
    unittest.main()
