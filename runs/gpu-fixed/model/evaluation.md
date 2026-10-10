# TextOrigin evaluation

Purpose: release_candidate

Dataset SHA-256: 6864f878323c1bd6e8e6f2b254d8af0fb6696798a1f21a5981f78d0a4c849ccd

| Document metric | Value |
|---|---|
| precision_human | 0.473313 |
| recall_human | 0.940000 |
| precision_ai_written | 0.224719 |
| recall_ai_written | 0.040000 |
| precision_ai_paraphrased | 0.925659 |
| recall_ai_paraphrased | 0.773547 |
| macro_f1 | 0.513437 |
| accuracy | 0.584390 |
| f1 | 0.513217 |
| f1_human | 0.629605 |
| f1_ai_written | 0.067912 |
| f1_ai_paraphrased | 0.842795 |
| brier_score | 0.591426 |
| expected_calibration_error | 0.266132 |

## Limitations

- AI-paraphrased labels come from one synthetic paraphrase model.
- Evaluation uses one dataset and does not establish cross-domain authorship accuracy.
- Model scores are uncalibrated.
- No guaranteed accuracy or quality threshold has been approved.

See evaluation.json for confusion matrices, error analysis, baselines and provenance.