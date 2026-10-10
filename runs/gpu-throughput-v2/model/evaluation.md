# TextOrigin evaluation

Purpose: release_candidate

Dataset SHA-256: 6864f878323c1bd6e8e6f2b254d8af0fb6696798a1f21a5981f78d0a4c849ccd

| Document metric | Value |
|---|---|
| precision_human | 0.911765 |
| recall_human | 0.744000 |
| precision_ai_written | 0.747994 |
| recall_ai_written | 0.932000 |
| precision_ai_paraphrased | 0.933761 |
| recall_ai_paraphrased | 0.875752 |
| macro_f1 | 0.851043 |
| accuracy | 0.850567 |
| f1 | 0.851008 |
| f1_human | 0.819383 |
| f1_ai_written | 0.829920 |
| f1_ai_paraphrased | 0.903826 |
| brier_score | 0.234869 |
| expected_calibration_error | 0.042229 |

## Limitations

- AI-paraphrased labels come from one synthetic paraphrase model.
- Evaluation uses one dataset and does not establish cross-domain authorship accuracy.
- Model scores are uncalibrated.
- No guaranteed accuracy or quality threshold has been approved.

See evaluation.json for confusion matrices, error analysis, baselines and provenance.