# Model cards

## Weekly sales forecasters

**Purpose:** compare one-week-ahead recorded-sales forecasts on a historical, training-selected UK SKU panel.

**Models:** last-week baseline; Random Forest (200 trees, depth 12); XGBoost (300 trees, depth 6, learning rate .05); fixed 50/50 average of the two fitted tree models.

**Inputs:** prior observed units (lags 1, 2, 4), two- and four-week shifted means, annual calendar phase, last observed price, and train-derived historical-high-sales flag. Country and SKU encoding is fitted with training only. The Dominican Republic interaction is relevant only to the demo; it is constant zero in the UK case.

**Evaluation:** three earlier temporal validation folds; final 16-week chronological holdout. Median imputation, encoders and high-sales thresholds are fitted within their training periods. Previous holdout weeks are observed before forecasting the next week.

**Metrics:** MAE, R² and WAPE in forecast_scores.csv; predictions and backtest folds are exported. Negative R² and models losing to the naive baseline remain in the reports.

**Limits:** 12 selected SKUs from one old retailer; sparse/intermittent sales and unusually large transactions; no inventory/stockout variables; returns excluded; historical prices are not a controlled experiment.

**Operational decision:** the recorded test does not justify deploying the ensemble as an improvement over the last-week baseline.

## Pricing analysis

Synthetic two-group experiment with 800 unique independent units. Welch mean-difference tests and 95% intervals; Holm correction for revenue, units and contribution margin. Margin omits overhead and other missing costs. The intervals in the chart are absolute-difference intervals scaled by the observed control mean, not ratio confidence intervals. The measured simulation does not establish a margin increase.

## Feedback classifier

**Data:** 650 bilingual template-generated texts; three synthetic labels. Negation and accents are preserved.

**Configurations:** TF-IDF + logistic regression (unigrams/bigrams), and MultinomialNB. Bigram logistic regression is the predetermined reporting model, not selected after seeing a test winner.

**Tests:** 80/20 stratified random holdout and a separate 20% template-family group holdout. Vectorizers fit only training text. No template families overlap in the second split.

**Metrics:** accuracy, macro-F1 and each class F1 in sentiment_simulation.csv; unseen-template confusion matrix and individual predictions are exported.

**Limits:** labels and wording are generated, ambiguity is simulated, languages are mixed, and no actual customer-review annotations are available. The large performance decline on unseen families prevents claiming readiness for real-world reviews.

