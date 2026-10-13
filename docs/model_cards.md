# Model cards

## Daily operations ensemble — real UCI data

**Purpose:** predict the next day's distinct retained UK order IDs for a historical capacity-planning exercise. This target differs from weekly SKU units.

**Inputs:** prior order counts at 1/2/7/14/28/364 days; shifted 7/28-day means; weekday, annual calendar harmonics, time trend and predetermined year-end/new-year flags. Weekday/season interactions use categories learned from training. No future revenue, price or order count enters the model.

**Models:** RF (200 trees, depth 10, minimum leaf 4); XGBoost (300 trees, depth 3, learning rate .05, minimum child weight 5, lambda 10); standardized Ridge (alpha 10). Two predetermined regularization configurations and 36 three-model weight combinations per configuration are compared with pooled MSE across three earlier 28-day folds. The chosen 80/10/10 weights are locked before final evaluation.

**Test:** 112 chronological days, 15 August–4 December 2011. Static models use observed prior days as they arrive; this is one-day forecasting, not a 112-day open-loop prediction. Final R² 0.844, MAE 9.64 orders/day, WAPE 15.76%; same-weekday baseline MAE 11.56. Ridge has marginally higher final R²; the ensemble has lower MAE. Test results are not used to change the blend.

**Limits:** historical one-retailer coverage; no actual staffing costs/capacity or business intervention; recorded date convention; substantial calendar contribution and uneven weekday scores. Saturday's constant zero target has undefined R². A current independent period is required before operational use.

**Explanations:** weighted tree-path SHAP plus additive standardized-linear contributions. Component predictions are floored at zero; the exact resulting correction is exported separately and included in reconstruction checks.

## Weekly demand ensemble — simulation

**Data:** original seed-42, 3,744-row generator, 36 country/category series. Promotion, holiday and market indices are assumed known before prediction; this is not real NielsenIQ data or work history.

**Inputs:** grouped shifted lags and rolling statistics, last observed price, training-derived high-sales flag, annual phase and series/season interactions. The Dominican Republic interaction is included only where that country occurs. Preprocessing and all thresholds fit only the applicable training period.

**Selection:** three earlier 8-week folds; 3 fixed regularization configurations and 45 weight combinations each (9 two-tree blends and 36 three-model blends), scored by pooled MSE. Selected configuration: smooth; RF depth 8/minimum leaf 12, XGBoost depth 2/child weight 10/lambda 20, Ridge alpha 30. Weights: 10/30/60.

**Final test:** 2022 holdout; R² 0.936, MAE 30.75 units, WAPE 6.37%. Median per-series R² is 0.795; 17 of 36 series exceed 0.80. Pure Ridge has lower final test MAE than the selected ensemble. Pooled R² includes between-series variation; it is not an accuracy percentage or proof of uniform quality. The generator is unchanged.

**Limits:** synthetic signal and assumed exogenous availability. The historical holdout was already inspected in the first revision; this is a retrospective rerun, not an independent new test.

## Weekly SKU research — real UCI data

12 SKU identities are frozen using observations before the first validation origin. Changing values after that origin cannot change the cohort. This corrects the initial retrospective cohort-selection bias, so cohort membership differs from the first version.

Three earlier 8-week folds select conservative regularization and 90/10/0 RF/XGBoost/Ridge weights. Final 16-week R² is 0.252, MAE 390.17 and WAPE 65.35%. The validation-selected 13-week mean has lower final MAE (304.61). This case does not meet the target and does not justify an ensemble deployment.

Unusually large transactions, variable individual-SKU sales and unobserved inventory/promotion activity limit this case. Zero recorded positive sales do not identify stockouts. Returns are audited and excluded. Results remain exported; daily-order scores are never substituted for SKU scores.

## Pricing simulation

800 independent generated units. Welch mean-difference intervals and Holm correction for revenue, units and contribution margin. Assumed costs are 45% of regular price; margin omits overhead. Chart intervals are absolute-difference intervals scaled by the observed control mean, not ratio confidence intervals. The run does not establish a margin increase or a real retailer profit impact. Missing/nonfinite outcomes, keys and costs are rejected explicitly.

## Feedback diagnostic

650 generated bilingual texts; three synthetic labels. Unique full texts can share template families. The same seed-42 random and family-group holdouts are retained; no family overlaps the group split.

A fixed word/character TF-IDF + balanced logistic-regression candidate improves group-held-out accuracy from the older bigram model's 19.7% to 29.3%. Its random-split accuracy is 89.2%. A training-majority baseline, unigrams, bigrams and Naive Bayes remain reported. Macro-F1 always includes all three classes.

This remains a **failed generalization diagnostic**. The generator includes ambiguous wording with labels assigned beforehand. The split also changes class composition. No NLP readiness is claimed; use human-labelled real text and language/domain evaluation before a real sentiment project.

## Metric definitions

R² is a regression score relative to a constant prediction of the evaluation mean; it can be negative. It is undefined for a constant target and recorded as missing. MAE retains the target's units. WAPE is aggregate absolute error divided by aggregate absolute target; it is undefined when that denominator is zero. Targets and aggregation levels must match before comparing numbers.

References: [scikit-learn R²](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html), [time-series validation](https://otexts.com/fpp3/toolbox.html).
