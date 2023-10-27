validate_dict = {
    "training_conf": {
        "training": {
            "data_config": {"function": ("check_file_exists",), "name": None},
            "metrics_config": {
                "function": (
                    "check_none",
                    "check_file_exists",
                ),
                "name": None,
            },
            "model_params": {
                "model_name": {
                    "function": ("check_if_equals_name",),
                    "name": ["poly"],
                },
                "n_bins_lda": {
                    "function": ("check_if_positive_int",),
                    "name": None,
                },
                "param_hparams": {
                    "random_seed": {
                        "function": (
                            "check_none",
                            "check_if_positive_int",
                        ),
                        "name": None,
                    },
                    "l2_param": {
                        "function": ("check_if_positive_float",),
                        "name": None,
                    },
                },
            },
            "training_hparams": {
                "batch_size": {
                    "function": ("check_if_positive_int",),
                    "name": None,
                },
                "multi_cycle_params": {
                    "total_cycles": {
                        "function": ("check_if_positive_int",),
                        "name": None,
                    },
                    "cycle_def": {
                        "function": ("check_if_equals_name",),
                        "name": [
                            "parametric",
                            "non-parametric",
                            "complete",
                            "only-non-parametric",
                            "only-parametric",
                        ],
                    },
                    "save_all_cycles": {
                        "function": ("check_if_bool",),
                        "name": None,
                    },
                },
            },
        }
    },
}
