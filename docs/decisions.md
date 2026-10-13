# Decisions I can explain in an interview

| Decision | Reason | Tradeoff |
|---|---|---|
| Daily-order capacity planning as the main real-data forecast | Invoice timestamps support a recorded workload target; SKU units are volatile and lack inventory context | It is a different business question from replenishment quantities |
| Preserve the unsuccessful SKU experiment | Model selection and a deployment decision should follow evidence | R² and MAE failures stay visible |
| Freeze SKU identities before validation | Prevent later validation popularity from influencing cohort choice | The revised cohort differs from the initial study |
| Tune configuration and weights with earlier folds | A fixed 50/50 average may not help; all test comparisons need independent selection | Development scores are optimistic selection scores, not an extra test |
| Compare stronger simple baselines | Complexity should improve measured errors on matching targets | A rolling mean still wins on SKU units |
| Ridge alongside RF/XGBoost | Standardized linear/calendar effects can generalize more smoothly than trees | An ensemble does not automatically beat every component on the final test |
| Keep synthetic generators intact | Preserve the four-week learning case and its reproducibility | Simulated performance does not establish real market effectiveness |
| Separate every source and forecasting target | Daily orders, SKU units, synthetic demand and text labels are distinct contracts | Their scores cannot be substituted or joined on broad country names |
| Export weekday and per-series results | Pooled R² can hide uneven performance | Some constant-target scores are undefined |
| Exact explanation reconstruction | Weighted components and nonnegative floors must match reported forecasts | The postprocessing correction is reported separately |
| Treat NLP as a failed generalization diagnostic | Word/character features help but unseen-family accuracy is still weak | Human-labelled real reviews are required for readiness claims |
| Explicit margin assumptions and multiplicity correction | Revenue lift alone does not establish profit or a margin improvement | Pricing remains a simulation |
| DuckDB and shared Python modules | SQL, CLI and notebooks reuse typed analytical data and tested functions | Users install the package before execution |

The next operational check is an independent current period with an agreed workload metric. SKU deployment needs inventory/promotions and further temporal evaluation; NLP needs annotated real text. These are future checks, not claimed outcomes.
