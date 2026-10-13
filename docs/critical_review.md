# Critical review of the first revision

## Corrections implemented

| Finding | Correction | Evidence |
|---|---|---|
| A cleaner repository did not produce a strong SKU forecaster | Added an explicit daily-order capacity-planning task suited to transaction timestamps, while retaining SKU failures | Real daily ensemble R² 0.844; SKU R² 0.252, separately reported |
| The 50/50 blend and tree settings were not selected from validation | Bounded regularization/weight search with strictly earlier temporal folds | Exported selection tables and future-target invariance tests |
| SKU identities were selected using weeks inside the earlier validation folds | Freeze the cohort before the first validation origin | Mutation test after the selection cutoff leaves SKU IDs unchanged |
| Only a weak last-week baseline was shown | Add 4/8/13/26-week means and a same-weekday daily baseline | Each score uses the same target and test rows within its case |
| Short lags missed longer calendar/history information | Add shifted long lags/rolling variation and series or weekday seasonal effects; compare Ridge | Features and preprocessing fit without final targets |
| The demo forecast chart mixed countries sharing a category name | Require one country/category curve; reject duplicate dates | Plot guard test and regenerated demo chart |
| Metrics hid uneven group/weekday behavior | Export per-series and per-weekday scores; mark constant-target R² undefined | forecast_by_series.csv / scores_by_weekday.csv |
| Missing finite-value/date/key checks could admit invalid data | Strengthen transaction, weekly, pricing and feedback contracts | Adversarial tests |
| Negative regression outputs are not meaningful order/unit counts | Floor component predictions at zero and audit the explanation correction | Nonnegative outputs and SHAP reconstruction test |
| High random-split NLP accuracy did not generalize | Add majority and word/character candidates; keep the same unseen-template holdout | 29.3% unseen-family accuracy remains labelled insufficient |
| Documentation and SHAP assumed a fixed two-model average | Explain selected RF/XGBoost/Ridge weights and update notebooks/cards | All reported component contributions reconstruct the selected forecast |

## Remaining limitations

The daily case predicts orders recorded per day, not inventory demand, SKU quantities, profit or unconstrained customer demand. Its high pooled R² includes calendar variation. The underlying source is 2009–2011 giftware data, not current supermarket activity. No production or staffing-savings claim follows from this historical test.

SKU performance remains weak and the fixed cohort is different from the first revision. Old/new SKU numbers cannot be marketed as a like-for-like improvement. The original simulation is preserved and labelled; its high pooled R² does not transfer to real SKU demand.

NLP needs a real annotated dataset. Ambiguous generated phrases and group composition limit the present exercise. Pricing still uses assumed costs and simulated assignments. No common transaction identity permits joining those scenarios to UCI sales.

The weekly holdout was previously inspected. All new model selection uses earlier folds, but independent later data remain necessary for a deployment claim. The daily operational target is explicitly documented as a separate research question.
