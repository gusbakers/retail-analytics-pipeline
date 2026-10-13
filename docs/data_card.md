# Data sources and contracts

## Real sales: UCI Online Retail II

- **Creator:** Daqing Chen. Citation: Chen, D. (2012), *Online Retail II*, UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CG6D.
- **Source:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The small bundled UCI sample is an attributed excerpt under this license.
- **Coverage:** 1 December 2009 to 9 December 2011; two Excel sheets; 1,067,371 transaction lines in the downloaded workbook.
- **Business:** one UK-based online giftware retailer. This is historical data from one retailer, not current market demand or NielsenIQ data.
- **Currency:** GBP. Dates/hours retain the source's recorded local convention; a timezone is not inferred.

| Original field | Normalized field | Meaning |
|---|---|---|
| Invoice | order_id | Invoice identifier; C-prefixed invoices are cancellations |
| StockCode | product_id | SKU identifier |
| Description | product_name | Product label; SKU is the fallback when missing |
| Quantity | quantity | Recorded line-item quantity |
| Price | unit_price | Recorded unit price in GBP |
| InvoiceDate | order_date | Recorded date and time |
| Customer ID | customer_id | Customer identifier; may be missing |
| Country | country | Country recorded on the transaction |

**order_dow** is derived from order_date with Monday=0; **order_hour** is the actual hour.
**gross_sales = quantity × unit_price** on retained positive sales. It is **not** net revenue or profit.

### Cleaning and interpretation

1. Normalize field names, validate required keys and numeric/date types.
2. Remove exact duplicate rows; this is an explicit analysis assumption, not proof that all repeated lines are erroneous.
3. Exclude missing required fields, C-prefixed cancellations, nonpositive quantities, and nonpositive prices.
4. Keep product codes matching five digits with up to two alphabetic suffix characters. This excludes postage/discount/fee codes and may exclude unusual legitimate nonstandard SKUs; the excluded count is reported.
5. Retain missing customer IDs for aggregate sales analysis; do not pretend to know the missing customer.
6. Record every exclusion count in reports/metrics/transaction_quality.csv.

Returns and free/nonproduct lines remain part of the original download, but are outside this forecasting target. The pipeline cannot infer stockouts, unit costs or latent demand from sales alone.

### Forecasting panel

- 104 complete covered calendar weeks, 7 December 2009 to 28 November 2011.
- United Kingdom only, 12 SKUs selected using volume and at least 26 active weeks **before the first validation origin** (28 February 2011).
- 1,248 SKU-week records. Zero means no retained paid sales recorded in a covered week; stock availability is unknown.
- Final 16-week holdout begins 15 August 2011. Four initial weeks per SKU are removed when constructing lag features.
- Price features use the last observed price. No future week's average price, promotion flag or retailer cost is invented.
- Target: weekly positive paid units sold, not unconstrained market demand.
- Three earlier 8-week temporal validation folds remain outside the final test. The cohort is fixed before those folds; it differs from the initial version selected using all final-training weeks.
- Long lags (8, 13, 26, 52), shifted rolling means/median/variation and country/category seasonal terms supplement short lags. Missing long history is imputed inside training; no future target is filled into a lag.

## Real daily operations target

This is a separate target from SKU demand: distinct retained paid UK order IDs within each recorded date, for an historical next-day workload-planning exercise. All UK SKUs enter the count; no validation/test-popularity selection occurs.

- 728 consecutive covered days, 7 December 2009–4 December 2011. The boundary incomplete weeks are excluded consistently with the weekly panel.
- Zero means no retained order was recorded that day. It does not prove a closure or the absence of unconstrained demand.
- 28 initial warm-up days are omitted. The final 112-day holdout starts 15 August 2011; 588 earlier feature rows train the final model.
- Three earlier 28-day validation folds start 23 May 2011 and end 14 August 2011.
- Lags 1/2/7/14/28/364, prior rolling means, weekday, annual calendar harmonics, time trend and calendar flags are available at forecast time. No future quantity, customer, price, revenue or order count is used.
- The target counts IDs within a recorded date; a source ID recorded on different dates can enter each date. Source dates retain the recorded convention.
- This observational target measures recorded paid-order workload. It does not estimate inventory replenishment quantities, current market demand or causal staffing savings.

## Independent learning simulations

These preserve methods from my four-week NielsenIQ-inspired case study. They are **not real NielsenIQ data, employment, internal messages, approvals, or measured production outcomes**.

| Source | Size / seed | Contract and assumptions |
|---|---|---|
| Demand demo | 3,744 rows, 104 weeks, 6 countries × 6 categories; seed 42 | Original Week 2 generator; 56 missing competitor-index values. Promotion, holiday and economic/competitor indices are assumed available before each demo prediction. |
| Pricing | 800 independent units, 400/group; seed 2427 | New enriched Week 4 sample. Treatment has 8% price discount and a simulated 22% volume multiplier; costs are assumed to be 45% of regular price. USD. |
| Feedback | 650 reviews, 273 positive / 215 negative / 162 neutral; seed 2240 | Bilingual templates with synthetic labels. Unique full texts can share a template family. Country/category labels are simulated, with no linkage to real retailer invoices. |

The Week 1 daily 3,020-row simulation is **not** concatenated with the different weekly panel. Its quality-audit methods are applied to the real source. The Week 3 revenue-only A/B sample is replaced by the enriched Week 4 experiment; its historical narrative lift is not reused.

No common transaction identity exists across UCI sales, pricing and feedback. Shared country/category names do not justify a row-level join or a causal sentiment-to-sales claim. Each module exports results with its own source label.

## Small bundled samples

- **data/samples/uci_transactions_sample.csv:** 300 rows sampled from the actual two-sheet UCI source with seed 42. Useful for schema inspection; not used to estimate full-dataset totals.
- **data/samples/grocery_transactions_demo.csv:** the original 30-row demo. It has no prices and is not the Instacart competition dataset. Kept as a historical reference, outside the new pipeline.
- Large raw data, DuckDB and regenerated processed tables are ignored by Git. The download URL, source/cache checksums, versions and execution configuration are recorded in reports/metrics/run.json.
