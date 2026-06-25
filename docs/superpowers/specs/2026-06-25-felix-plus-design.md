# FELIX++ Edit-Based VSL Glossing Design

## Context

The project already evaluates two strong text-to-text systems for Vietnamese to
VSL gloss generation:

- `vit5_base_15ep`: ViT5-base fine-tuned on the project split.
- `ensemble_mbr`: candidate-pool MBR over ViT5-base and BARTpho-syllable.

The current best system is `ensemble_mbr` with BLEU 97.00, chrF 98.53, WER 1.89,
TER 1.66, and EM 93.36 on the shared 949-example test set. The third method
should be methodologically distinct from these seq2seq/ensemble systems while
maximizing the chance of surpassing the current best leaderboard score.

The corpus analysis strongly supports an edit-based approach. On the current
test split, 98.31% of examples can be represented by deleting source tokens and
reordering the kept source tokens. The existing oracle report gives a
tag+pointer-only ceiling of BLEU 99.25 and WER 0.539, and a
tag+pointer+insertion ceiling of BLEU 99.633 and WER 0.298. FELIX++ should target
this headroom.

## Goal

Build FELIX++ as a third experimental method:

- It must be an independent edit-based model, not another seq2seq backbone.
- It must train, validate, and test on the exact same split as ViT5-base and
  ViT5+BARTpho+MBR.
- It should prioritize leaderboard improvement, especially WER and EM, while
  preserving a clear research story for the report.

The objective is not to claim guaranteed superiority before training. The
objective is to maximize the chance of beating `ensemble_mbr` under the same
data split and metrics.

## Non-Goals

- Do not create a new train/validation/test split.
- Do not use test labels for tuning, reranking, threshold selection, or
  post-processing decisions.
- Do not make FELIX++ a wrapper that chooses between ViT5/BARTpho predictions at
  test time. ViT5/BARTpho outputs may be used for error analysis or optional
  training-time distillation, but the main FELIX++ test prediction must decode
  from the source sentence through the edit model.
- Do not replace the existing leaderboard/evaluation code.

## Required Split Fairness

FELIX++ must use these existing files:

- `data/splits/train.jsonl`: 7,586 examples
- `data/splits/val.jsonl`: 942 examples
- `data/splits/test.jsonl`: 949 examples

Training reads only `train.jsonl`. Model selection, threshold tuning, reranker
weight tuning, and ablation decisions read only `val.jsonl`. Final reporting
decodes `test.jsonl` once with the chosen validation configuration.

The prediction file must preserve the existing schema:

```json
{"id": "...", "vie": "...", "vsl": "...", "category": "...", "pred": "..."}
```

Before scoring, the pipeline must verify:

- `predictions_test.jsonl` has exactly 949 rows.
- Its `id` sequence exactly matches `data/splits/test.jsonl`.
- No test source sentence is dropped.
- The same `vsl_gloss.evaluate.score_file` path is used for metrics and
  category breakdowns.

## Recommended Approach

Use a FELIX-style edit model with three prediction components:

1. **Tagging head** predicts whether each source token is kept or deleted.
2. **Pointer/reorder head** predicts the target order of kept source tokens.
3. **Insertion head** fills tokens or short phrases that do not appear in the
   source.

This follows the core idea of FELIX: split generation into tagging/reordering
and insertion. It also borrows from LaserTagger's keep/delete/add-phrase framing
and Levenshtein Transformer's view of generation as explicit edit operations.

## Architecture

FELIX++ should extend the existing `vsl_gloss.felix` package rather than start a
parallel implementation from scratch.

### Encoder

Run an encoder sweep because the leaderboard margin is tiny:

- Primary high-capacity candidate: `xlm-roberta-large`.
- Vietnamese-focused candidate: `vinai/phobert-base-v2`.
- Control candidate: current `xlm-roberta-base`.

The selected encoder should be chosen by validation WER first, then BLEU/EM.

### Tagging Head

For each source token, predict `KEEP` or `DELETE`. Labels are produced by
case-folded greedy alignment between target tokens and source tokens, matching
the existing FELIX label extractor. Duplicate source tokens must be matched to
distinct positions through per-token queues.

### Pointer/Reorder Head

Predict successor links over `[BOS, source tokens] -> [source tokens, EOS]`.
During training, deleted tokens are masked out as invalid successor targets.
During inference, the model should support multiple pointer candidates instead
of a single greedy path so the reranker can recover from local reorder mistakes.

### Insertion Head

Represent target tokens not aligned to source tokens as insertions at slots
between ordered kept tokens. The insertion component should support:

- no insertion
- punctuation insertion or punctuation replacement
- one-token lexical insertion
- short phrase insertion observed in training

The simplest high-yield design is a slot-level insertion classifier over a
closed phrase vocabulary mined from `train.jsonl`, with a reserved `NONE` class.
The phrase vocabulary should include insertion phrases above a frequency
threshold and all punctuation repair labels needed by the training split. The
initial threshold should be `min_count = 1` because insertion examples are rare;
if the resulting vocabulary becomes unexpectedly large, cap it by validation
WER rather than test performance.

If validation shows the phrase vocabulary is too small, a second-stage masked LM
filler can be added later, but the first implementation should keep insertion
deterministic and easy to ablate.

### Format Head And Normalization Layer

Use a small auxiliary format head for final punctuation and casing decisions.
Its labels are derived from `train.jsonl` by comparing rendered edit outputs
against target glosses. It should predict only compact, high-impact decisions:

- final punctuation class
- whether to preserve, lowercase, or uppercase the first alphabetic token
- whether punctuation spacing requires repair

This head provides `format_loss` during training. After the model emits text,
the existing normalization utilities still run as a deterministic final cleanup.

### Deterministic Normalization Layer

Before scoring, normalize spacing and punctuation with the project's existing
normalization utilities. Avoid hard-coded truecasing rules that blindly
capitalize every sentence, because current FELIX errors show that casing alone
can turn otherwise correct predictions into mismatches.

Formatting decisions should be validation-tuned:

- preserve source casing when target examples in the same pattern preserve it
- repair final punctuation only when confidence is high
- apply punctuation spacing normalization consistently

## Training Strategy

Train with a multi-task objective:

```text
loss = tag_loss
     + lambda_pointer * pointer_loss
     + lambda_insert * insertion_loss
     + lambda_format * format_loss
```

The default validation target is lowest WER. BLEU and EM are tie-breakers.

Use staged training:

1. Train tag head and encoder until deletion behavior stabilizes.
2. Add pointer loss and train reorder behavior.
3. Add insertion loss and train lexical/punctuation repairs.
4. Jointly fine-tune all heads with category-balanced sampling or loss weights.

Category weighting should upweight rare but important categories:

- `reorder_only`
- `deletion_reorder`
- `lexical`

This is important because the current ViT5/MBR errors concentrate in deletion,
reorder, lexical, and formatting edge cases, while most examples are simpler
deletion-only or identical cases.

Optional training-time distillation is allowed if it is kept clean:

- Teacher outputs may be generated for `train`; `val` teacher outputs may be
  generated only for diagnostics and error analysis.
- Teacher signals may be used as auxiliary consistency features or losses on
  `train` only.
- The final FELIX++ prediction on `test` must not choose among ViT5/BARTpho test
  outputs.

## Decoding And Reranking

Do not rely on greedy decode alone. FELIX++ should generate an internal
candidate set per source sentence:

- several tag candidates when token probabilities are uncertain
- several pointer paths for reorder ambiguity
- several insertion candidates per slot
- variants with strict and relaxed punctuation repair

Then select one prediction using a validation-tuned reranker.

Recommended reranker features:

- tag, pointer, and insertion log probabilities
- length ratio between prediction and source
- number of deleted tokens
- number of inserted tokens
- source-token coverage
- repeated-token penalty
- punctuation consistency
- category-aware priors learned from the training labels
- internal consensus score among FELIX++ candidates, similar in spirit to MBR

The reranker must be trained or tuned on `val.jsonl` only. Once tuned, freeze the
weights and decode `test.jsonl` with the fixed configuration.

## Evaluation Plan

Main systems to report:

- `vit5_base_15ep`
- `bartpho_syllable`
- `ensemble_mbr`
- existing `felix`
- `felix_plus_tag_pointer`
- `felix_plus_insert`
- `felix_plus_rerank`
- optional `felix_plus_large_encoder`

Metrics:

- BLEU
- chrF
- WER
- TER
- Exact Match

Report both overall metrics and category metrics for:

- `identical`
- `deletion_only`
- `reorder_only`
- `deletion_reorder`
- `lexical`

Required ablations:

- without insertion head
- without reranker
- without category weighting
- encoder comparison
- greedy decode versus candidate reranking

## Success Criteria

Minimum success:

- FELIX++ beats the existing `felix` baseline by a large margin.
- The report clearly explains why edit-based modeling is different from
  seq2seq/MBR.

Strong success:

- FELIX++ approaches or beats `vit5_base_15ep` on WER or EM.

Ambitious success:

- FELIX++ beats `ensemble_mbr` on at least one primary leaderboard metric, with
  WER and EM prioritized.

Even if the ambitious target is not reached, the method remains valuable if it
shows category-specific gains in deletion/reorder/lexical cases and has a clear
oracle-backed explanation.

## Risks And Mitigations

**Risk: pointer decoding remains weaker than seq2seq generation.**
Mitigation: use candidate pointer paths and reranking instead of greedy pointer
chasing.

**Risk: insertion labels are sparse.**
Mitigation: use a closed phrase vocabulary mined from train and report insertion
as an ablation. Keep a future path open for masked LM insertion if validation
requires it.

**Risk: formatting changes hurt exact match.**
Mitigation: tune formatting on validation and preserve source casing unless the
model or validation rules support a change.

**Risk: leaderboard gain is small because ViT5/MBR is already near ceiling.**
Mitigation: focus on WER/EM and per-category analysis. Use the oracle ceiling to
argue that edit modeling has headroom even when final aggregate gains are small.

## References

- FELIX: Flexible Text Editing Through Tagging and Insertion:
  https://aclanthology.org/2020.findings-emnlp.111/
- LaserTagger / Encode, Tag, Realize: High-Precision Text Editing:
  https://aclanthology.org/D19-1510/
- Levenshtein Transformer:
  https://proceedings.neurips.cc/paper/2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html
- Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine
  Translation:
  https://aclanthology.org/2020.coling-main.398/
