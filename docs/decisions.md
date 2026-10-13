# Decisions I can explain in an interview

| Decision | Reason | Tradeoff |
|---|---|---|
| Replace the tiny demo as the main source | A public, attributed transaction history supports an actual sales story and realistic data-quality decisions | Historical giftware data does not describe today's retail market |
| Keep the four topics in one repository | Quality, forecasting, experimental analysis and NLP demonstrate a coherent analytical workflow | The pricing and NLP datasets are independent simulations, not a merged customer panel |
| DuckDB for analytical SQL | Query the full pandas sales table and persist typed analytical results without a database server | This replaces the small demo's SQLite path |
| Shared functions in src/market_insights | Notebook explanations and main.py use the same code | Users install the package once before running |
| Train-only SKU selection | Avoid using final-holdout popularity to choose the products being forecast | Conclusions apply to 12 training-selected UK SKUs, not every product |
| One-week horizon with updated observations | Matches forecasting a new week after the previous week's actual sales arrive | It does not simulate forecasting all 16 weeks from one fixed origin |
| Last-week baseline | A complex algorithm should earn its extra complexity | In this recorded holdout the baseline has the lowest MAE |
| Fixed model settings and ensemble weights | Preserve a clear reference comparison without tuning on final test | No claim that the ensemble is optimal |
| Explicit zero-sales semantics | Complete export coverage allows recording no paid sales for a SKU/week | Zero sales cannot tell us whether inventory was available |
| Costs and margins labelled simulation | Revenue does not alone establish profitability | Assumed costs do not represent the real retailer |
| Unseen-template NLP evaluation | Random splitting of templated texts can look much better than generalization | A template holdout is still not a test on human-labelled real reviews |
| SHAP training-path reference | Provides additive tree-model explanations; averaged contributions must reconstruct the ensemble | Attribution is associative, not causal |
| Code MIT, UCI excerpt CC BY 4.0 | Code and source data have separate licensing and attribution | Data-source conditions remain applicable |

## What I would investigate next

- Why the last-week baseline wins and which SKUs/weeks drive the largest error.
- Forecast stability with inventory availability and subsequent time periods.
- Returns separately from positive paid sales, rather than claiming net profit.
- New human-labelled feedback and language/domain-specific evaluation.
- Pricing with actual randomization, complete unit economics and a predetermined primary outcome.

These are future investigations. They are not claimed as completed business impacts.

