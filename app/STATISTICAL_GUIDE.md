# Reading TrendAnalysis statistical panels

The terminal is a research aid. No single coefficient is universally good or bad, and no plot here is a buy/sell signal. Read each result alongside its units, selected dates, valid/matched observation count, uncertainty, missing-data notes, and stability over time.

## Distribution & variability

Choose all selected series or a subset. Every selected variable receives its own chart and summary; unrelated raw units are not placed on one shared numerical axis.

- **Observed values:** original recorded levels. Useful for ranges and clustering, but trending levels can make the distribution mainly reflect the trend.
- **Absolute change:** current observation minus the previous available observation. Useful for size and direction in the variable's own units, including signed values that cross zero.
- **Percentage change:** `(current / previous - 1) × 100`, only when the previous value is positive. Useful for ordinary price returns, but inappropriate for negative/zero bases and not automatically comparable across different observation frequencies.

Visuals:

- **Histogram:** each bar counts observations inside a numerical range. Bar height is not the probability of a future move; shape depends on the bin width.
- **Box plot:** median, middle 50% of observations, and points flagged as potential outliers under the plotted convention. An outlier is not automatically a data error or trading opportunity.
- **Cumulative percentile:** at each x-value, the y-value is the share of observations at or below that value. It describes the chosen historical sample, not a calibrated forecast.

Numbers:

- **Mean:** arithmetic average; sensitive to extreme values.
- **Median:** middle observation; less sensitive to extremes.
- **Standard deviation:** spread around the mean in the metric's units.
- **25th–75th percentiles:** middle half of observed values.
- **5th–95th percentiles:** central 90% observed range.
- **Skewness:** distribution asymmetry; positive values often reflect a longer/more influential upper tail, negative values a lower tail.
- **Excess kurtosis:** tail/peakedness descriptor relative to a normal-distribution reference. Sensitive to extremes and requires a reasonable sample.

Plain-language descriptions are rule-based and descriptive. Fewer than 30 observations triggers a caution, not a universal validity threshold. Large samples can still be biased, serially dependent or unrepresentative.

## Relationships

Relationship calculations first intersect the two raw series on exact timestamps, then form changes on that matched observation grid. This avoids comparing changes formed over unequal observation spans when either series has missing bars. It does not automatically resample mismatched markets; exact timestamps and compatible sampling cadence are still required.

### Common comparison scales

- **Log return (%):** `100 × ln(Pₜ / Pₜ₋W)`. Default for positive price series. It is scale-free and symmetric for rises/falls of similar proportional size. Zero or negative prices are excluded.
- **Z-score of log return:** each log return is centered and divided by its study-period standard deviation; axes are in standard-deviation units.
- **Percentage change (%):** `100 × (Pₜ / Pₜ₋W − 1)`. Requires a positive prior value.
- **Z-score of percentage change:** percentage changes standardized to study-period standard deviations.
- **Absolute change:** `Xₜ − Xₜ₋W`. Valid for signed values but remains in native units, so slopes are not comparable across different units.
- **Z-score of absolute change:** absolute changes standardized within the study period. Useful for comparing unusual movement size across series, but does not turn the result into a causal or predictive relationship.

The optional indexed-level chart rebases each positive series to 100 at the first matched timestamp. It compares relative path, not statistical significance. Z-scoring changes the units of plotted movements; Pearson and Spearman correlation are invariant to positive linear rescaling.

### Correlation strength and GREEN/RED status

Conventional Cohen-style magnitude bands for absolute Pearson correlation are:

| Absolute Pearson r | Strength |
|---:|---|
| `< 0.10` | Negligible |
| `0.10–0.29` | Small |
| `0.30–0.49` | Moderate |
| `≥ 0.50` | Large |

These are broad conventions, not universal scientific cutoffs. A stronger relationship is not automatically economically useful.

- **GREEN:** `|r| ≥ 0.30`, approximate 95% interval excludes zero, Bonferroni-adjusted Fisher-z p-value `≤ 0.05`, at least 60 matched observations, and effective sample size `≥ 30`.
- **RED:** enough data to evaluate the relationship, but one or more strength/significance gates fail.
- **GREY:** insufficient observations or an interval/p-value cannot be estimated responsibly.

The app adjusts the selected pair's approximate p-value for the number of possible pairs in the loaded workspace. The interval and p-value use a Fisher-z approximation and Bartlett-style effective sample size that discounts some serial dependence. This is not a full heteroskedasticity/autocorrelation-consistent test and does not fix non-stationarity, outliers, or data-source mismatch. Treat GREEN as evidence under the displayed assumptions, not as a buy/sell signal.

- **Pearson r:** linear association between changes, from −1 to +1.
- **Spearman rho:** rank-based monotonic association.
- **OLS slope:** depends on the chosen transformation and units; the correlation is usually the better cross-variable comparison statistic.
- **Rolling correlation:** repeats the correlation within a moving window. The approximate intervals are pointwise, not simultaneous, and adjacent windows overlap heavily.
- **Correlation heatmap:** each pair uses its own exact matched timestamp grid. Hover for matched observation count; heatmap colors are descriptive and do not denote significance.

## Event response

The app marks dates when the main variable's change over the selected window reaches the positive or negative threshold, then measures target response after 1 or more target observations. Forward target data after the study end date is used only to measure realized responses; late events can lack complete outcomes.

- **Event mean/median:** average/middle target response after dates meeting the threshold.
- **Non-event baseline:** response on matched dates where neither event threshold was crossed.
- **Event minus baseline:** difference between the event mean and baseline mean, expressed in the target response units. When response is a percentage return, the difference is measured in percentage points.
- **Events found vs responses available:** these counts differ when future target observations are not yet available or are missing.

The added difference graph focuses on the quantity of interest: event response minus ordinary baseline. Its intervals use a **circular moving-block bootstrap** over chronologically matched observations. Blocks try to preserve local dependence and some dependence from overlapping forward horizons. Block length is limited by sample size; the interval is unavailable for small/sparse samples or too few valid replicates. This is more appropriate than treating every event as independent, but it remains exploratory: block length is approximate and thresholds/horizons tried after viewing results are not corrected for multiple testing. An interval crossing zero means this sample does not clearly distinguish the effect estimate from zero; an interval excluding zero is a candidate association, not proof of causation.

## Lead/lag

The graph aligns measured changes at multiple offsets. Negative lag means the comparison series is aligned earlier than the reference under this app's convention; positive means the comparison series is later. Offsets count exact matched observations. The UI also reports an approximate elapsed time, calculated as offset observations multiplied by the median interval between matched timestamps; missing bars and session gaps can make this approximate rather than exact.

For a selected change window of W observations, the app tests offsets spaced by W observations (for example, W=5 tests −10, −5, 0, +5, +10 when testing two windows in either direction). This reduces direct overlap between adjacent change windows, but it cannot eliminate serial dependence, regime changes, or timing irregularities in data with different frequencies or missing dates. The displayed offset and window length must be read together.

Pointwise confidence intervals use a moving-block bootstrap of paired changes. This resamples contiguous blocks and approximately preserves local dependence; the block length is chosen from the data length and change window. Family-wise adjusted p-values come from a circular-shift maximum-statistic test: the comparison series is shifted relative to the primary series, shifts close to the tested offsets are excluded, and each observed lag is compared with the maximum absolute correlation across all tested lags under each shifted alignment. This helps account for choosing the strongest of the tested lags while preserving each series' broad serial structure approximately.

The circular-shift test assumes the aligned change process is sufficiently stationary for shifts to be meaningful. It does not correct for testing many different variable pairs, windows, thresholds or study periods. The p-value can be unavailable for short samples or wide lag ranges; the app should say so instead of manufacturing a result. A peak that survives this test remains an association, not proof of prediction or cause. Confirm candidates on a pre-specified untouched chronological period.

## How to judge a result

Do not label a result good or bad from one coefficient. Inspect:

1. **Magnitude:** is the effect large enough to matter in the relevant units?
2. **Uncertainty:** how wide is the interval, and what assumptions support it?
3. **Support:** how many valid observations and events are present?
4. **Stability:** does the relationship persist across dates and regimes?
5. **Incremental value:** does it add information over an appropriate ordinary-response baseline?

The current UI is exploratory. Lead/lag uses non-overlapping tested offsets, moving-block-bootstrap intervals and circular-shift max-statistic p-values adjusted over tested offsets, then conservatively adjusted across selected comparison variables. Same-period relationship p-values and intervals use the approximate effective-sample-size method described above. Event-response differences use moving-block bootstrap intervals. None of these methods corrects every form of dependence or all analyst choices; confirm candidate results on a pre-specified untouched chronological period before making claims of predictive or trading value.


## Multiple variables in event and lead/lag analyses

The event-defining variable/reference variable is a role in a particular research question, not a restriction to the workspace's main series. Event Study allows one selected event definition to be compared against multiple response targets. Lead/Lag allows one selected reference series to be compared against multiple series in one run. Each target/comparison remains a distinct result; it is not collapsed into a single pooled outcome. Lead/lag p-values are adjusted across offsets within each pair and then conservatively adjusted across the selected comparison set. Event-study intervals are not automatically corrected for selecting many event targets, thresholds or horizons, so treat multi-target results as exploratory and pre-specify the key comparisons when possible.
