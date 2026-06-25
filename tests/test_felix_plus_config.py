from vsl_gloss.config import Config


def test_felix_plus_config_defaults_are_available():
    cfg = Config.from_dict({"experiment_name": "felix_plus"})

    assert cfg.felix_plus.encoder_name == "xlm-roberta-base"
    assert cfg.felix_plus.max_source_length == 128
    assert cfg.felix_plus.insertion_min_count == 1
    assert cfg.felix_plus.max_candidates == 32
    assert cfg.felix_plus.selection_metric == "wer"
    assert cfg.felix_plus.category_weights["lexical"] > cfg.felix_plus.category_weights["identical"]


def test_felix_plus_yaml_overrides_nested_values():
    cfg = Config.from_dict(
        {
            "felix_plus": {
                "encoder_name": "xlm-roberta-large",
                "max_candidates": 64,
                "category_weights": {"identical": 1.0, "lexical": 6.0},
            }
        }
    )

    assert cfg.felix_plus.encoder_name == "xlm-roberta-large"
    assert cfg.felix_plus.max_candidates == 64
    assert cfg.felix_plus.category_weights["lexical"] == 6.0
