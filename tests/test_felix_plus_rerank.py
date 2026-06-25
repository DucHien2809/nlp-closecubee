from vsl_gloss.felix.plus_decode import EditCandidate
from vsl_gloss.felix.rerank import RerankWeights, choose_candidate, tune_weights


def test_choose_candidate_uses_weighted_features():
    candidates = [
        EditCandidate("A", [], {}, "final=NONE|case=preserve|spacing=1", {"model": -2.0, "repeat": 0.0}),
        EditCandidate("B", [], {}, "final=NONE|case=preserve|spacing=1", {"model": -0.1, "repeat": 1.0}),
    ]
    weights = RerankWeights({"model": 1.0, "repeat": -2.0})

    assert choose_candidate(candidates, weights).text == "A"


def test_tune_weights_prefers_lower_validation_wer():
    candidate_groups = [
        [
            EditCandidate("Tôi đau đầu .", [], {}, "final=.|case=preserve|spacing=1", {"model": -1.0}),
            EditCandidate("Tôi đau đầu", [], {}, "final=NONE|case=preserve|spacing=1", {"model": -0.1}),
        ]
    ]
    refs = ["Tôi đau đầu ."]

    weights = tune_weights(candidate_groups, refs, feature_names=["model"], grid=[-1.0, 1.0])
    assert choose_candidate(candidate_groups[0], weights).text == "Tôi đau đầu ."
