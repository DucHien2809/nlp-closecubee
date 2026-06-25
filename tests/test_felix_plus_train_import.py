from vsl_gloss.felix.plus_train import category_sample_weights, loss_weights_for_epoch, metric_is_better


def test_loss_weights_for_epoch_stages_training():
    assert loss_weights_for_epoch(epoch=1, tag_warmup_epochs=2, pointer_warmup_epochs=2) == {
        "tag": 1.0,
        "pointer": 0.0,
        "insertion": 0.0,
        "format": 0.0,
    }
    assert loss_weights_for_epoch(epoch=3, tag_warmup_epochs=2, pointer_warmup_epochs=2)["pointer"] == 1.0
    assert loss_weights_for_epoch(epoch=5, tag_warmup_epochs=2, pointer_warmup_epochs=2)["insertion"] == 1.0


def test_metric_is_better_uses_lower_wer():
    assert metric_is_better({"wer": 1.8}, {"wer": 2.0}, "wer")
    assert not metric_is_better({"wer": 2.1}, {"wer": 2.0}, "wer")
    assert metric_is_better({"bleu": 97.1}, {"bleu": 96.9}, "bleu")


def test_category_sample_weights_upweight_rare_categories():
    dataset = type(
        "Dataset",
        (),
        {"examples": [{"category": "identical"}, {"category": "lexical"}]},
    )()

    weights = category_sample_weights(dataset, {"identical": 1.0, "lexical": 5.0})

    assert weights == [1.0, 5.0]
