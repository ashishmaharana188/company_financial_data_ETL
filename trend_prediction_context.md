# Trend Prediction System Context

## Purpose

This file is a compact context-hydration document for future work on the financial trend prediction system built around the `company_financial_data_ETL` repository.

Use this file when a later task requires reconstructing the current architecture, engine responsibilities, data domains, and known modelling constraints.

Repository:
`https://github.com/ashishmaharana188/company_financial_data_ETL`

---

## Existing Data Foundation

The repository already contains multiple analytical data domains.

### Market / F&O

The current market layer includes cash, futures, options and related derived market information through `unified_market_master` and its materialized analytical tables.

Relevant derived structures include:

- `mv_options_aggregates`
- `mv_spot_futures_basis`
- `mv_unified_market_matrix`

Typical variables include:

- OHLC / close
- volume
- delivery percentage
- futures price
- futures open interest
- futures basis / cost of carry related measures
- option open interest
- option volume
- OI PCR
- volume PCR
- changes in derivatives positioning

### Macro / Global

The repository has daily and intraday macro/global ledgers.

Known macro/global series include:

- `Broad_Commodity`
- `US_VIX`
- `India_10Y_Yield`
- `US_10Y_Yield`
- `Brent_Crude`
- `Nifty_50`
- `US_Dollar_Index`
- `India_VIX`
- `Yield_Spread`
- `USD_INR`
- `India_CPI`
- other F&O / MCX / global asset data available in the repository

Relevant tables:

- `macro_daily_ledger`
- `macro_intraday_ledger`
- `global_assets_daily`
- `global_assets_intraday`

### Institutional / Trade Events

The repository has:

- `institutional_ledger`
- `trade_events_ledger`
- `mv_institutional_flow`

These support FII/DII and participant positioning, plus block/bulk-style event analysis.

### Financial Statements

The repository has scraped financial statements and related financial data, including:

- quarterly income statement
- yearly income statement
- quarterly balance sheet
- yearly balance sheet
- quarterly cash flow
- yearly cash flow
- yearly indirect cash flow
- raw financial data

Typical fundamental variables include:

- revenue
- profit / net income
- margins
- assets
- debt
- equity
- operating cash flow
- capex
- working capital
- valuation-related financial ratios / derived metrics

### Prediction Infrastructure Already Present

`prediction_ledger` already exists and is intended to store model outputs.

Known output fields include:

- `engine_name`
- `ticker`
- `asof_date`
- `horizon`
- `signal`
- `score`
- `confidence`
- `veto_flag`
- `penalty`
- `target_metric`
- `reason_json`
- `feature_json`

An existing `olsEngine1.py` also performs historical prediction and writes into this ledger.

---

## Important Data-Domain Separation

Do not treat the financial database as the complete knowledge base.

There are separate domains:

```text
NEWS DATA
    event facts / announcements / narrative

FINANCIAL DATA
    revenue / profit / assets / debt / cash flow / valuation

COMPANY OPERATIONAL DATA
    capacity / production / utilisation / order book / projects / commissioning

INDUSTRY OPERATIONAL DATA
    industry capacity / demand / production / competitors / market share / utilisation

MARKET / MACRO DATA
    price / F&O / flows / commodities / rates / FX / volatility / global markets
```

A news event should determine which external domains are required for contextualisation.

Example:

```text
News:
₹4,000 crore investment + 2.2 GW new capacity

May require:
- financial data: revenue, assets, CFO, historical capex
- company operational data: existing capacity, utilisation, project pipeline
- industry data: industry capacity, competitor capacity, demand
- market data: price, valuation, positioning
```

Operational capacity and peer capacity are **not financial data**.

---

# Revised Analytical Engine Architecture

The previous 10-engine proposal is retained conceptually but revised so that each engine has one distinct analytical responsibility.

## Engine 1: Market Microstructure Engine

### Core question
What is the stock/index doing internally right now?

### Inputs

- price
- OHLC
- volume
- delivery
- volatility
- futures basis where appropriate

### Main calculations

- trend
- momentum
- volume abnormality
- delivery abnormality
- volatility expansion/contraction
- price-volume confirmation
- short-term structure

### Output

- `direction_score`
- `trend_strength`
- `market_structure_state`

### Notes

The current `olsEngine1.py` belongs primarily in this engine family. It should be improved through proper out-of-sample and walk-forward validation.

---

## Engine 2: Derivatives Positioning Engine

### Core question
What do futures and options positioning imply about near-term market structure?

### Inputs

- futures OI
- futures price
- basis
- call OI
- put OI
- change in OI
- call/put volume
- OI PCR
- volume PCR
- participant-wise derivative positions

### Main calculations

Classify combinations such as:

- price up + OI up
- price down + OI up
- price up + OI down
- price down + OI down

Also evaluate option-chain positioning rather than treating PCR alone as directional truth.

### Output

- `derivative_bias`
- `positioning_strength`
- `support_resistance_structure`
- `positioning_change`

### Notes

PCR changes do not by themselves identify put buying, put unwinding, call buying, or call writing.

---

## Engine 3: Institutional Flow Engine

### Core question
Is larger capital accumulating, distributing, hedging, or remaining neutral?

### Inputs

- FII flow
- DII flow
- participant-wise futures positioning
- participant-wise options positioning
- block deals
- bulk deals
- trade events

### Main calculations

- net institutional flow
- flow acceleration
- flow persistence
- cash-versus-derivative divergence
- block/bulk participation
- accumulation/distribution patterns

### Output

- `institutional_bias`
- `institutional_strength`
- `accumulation_distribution_state`

### Notes

High delivery is not equivalent to institutional accumulation. Buyer identity requires stronger evidence.

---

## Engine 4: Global + Macro Regime Engine

### Core question
What macro environment is the Indian market operating inside?

### Inputs

- US VIX
- India VIX
- US 10Y
- India 10Y
- yield spread
- DXY / US Dollar Index
- USD_INR
- Brent / commodities
- global equity indices
- CPI
- other available macro/global series

### Main calculations

- risk-on / risk-off state
- volatility regime
- rate regime
- currency pressure
- commodity pressure
- global equity regime
- inflation regime

### Output

- `macro_regime`
- `macro_score`
- `risk_environment`

### Notes

This engine should provide context and condition other models. Avoid crude permanent rules such as a universal VIX threshold that automatically vetoes everything.

---

## Engine 5: Commodity-to-Sector Transmission Engine

### Core question
Which sectors and companies are economically exposed to the current commodity move?

### Inputs

- Brent
- crude and other MCX commodities
- copper
- steel
- other relevant commodity series
- CPI
- company sector
- financial history
- company operational data when available

### Main calculations

- commodity shock
- sector exposure
- historical sensitivity
- company sensitivity

### Output

- `sector_impact`
- `company_exposure`
- `commodity_sensitivity`

### Notes

Do not claim exact margin impact without the necessary exposure data, procurement mix, pass-through, hedging, inventory lag, and other operating inputs.

---

## Engine 6: Fundamental Regime Engine

### Core question
Is the underlying business improving, deteriorating, or remaining stable?

### Inputs

- income statements
- balance sheets
- cash flows
- earnings growth
- margins
- debt
- ROE / ROCE
- capex
- related financial quality metrics

### Main calculations

- earnings trend
- growth acceleration
- margin trend
- cash-flow quality
- balance-sheet stress
- capex intensity
- financial quality

### Output

- `fundamental_state`
- `fundamental_strength`
- `earnings_direction`
- `financial_risk`

### Notes

Fundamentals are slow-moving state variables. They should not be treated as direct predictors of very short intraday moves.

---

## Engine 7: Valuation + Interest-Rate Engine

### Core question
Given company fundamentals and the current rate environment, how stretched or depressed is valuation?

### Inputs

- P/E
- P/B
- FCF yield
- historical valuation
- sector valuation
- India 10Y
- US 10Y
- yield spread
- earnings growth

### Main calculations

- relative valuation
- historical valuation percentile
- sector-relative valuation
- rate-adjusted valuation context

### Output

- `valuation_state`
- `valuation_percentile`
- `valuation_risk`

### Notes

Historical valuation distributions are company- and sector-dependent. Avoid treating a single 5-year standard-deviation band as a universal rule.

---

## Engine 8: News + Operational Event Engine

### Core question
What changed in the real business, and how significant is the change relative to the company and industry?

### News inputs

- investment announcements
- capacity additions
- orders
- contracts
- acquisitions
- shutdowns
- regulatory actions
- management guidance
- project announcements

### Company operational inputs

- installed capacity
- production
- utilisation
- order book
- project pipeline
- commissioning

### Industry operational inputs

- industry capacity
- demand
- production
- competitor capacity
- market growth
- utilisation

### Financial contextual inputs

- revenue
- assets
- cash flow
- debt
- capex
- profit

### Main calculations

- event type
- event magnitude
- company significance
- industry significance
- execution dependency
- historical unusualness when data permits

### Output

- `event_type`
- `event_magnitude`
- `company_significance`
- `industry_significance`
- `execution_dependency`

### Notes

Do not reduce news to a simple positive/negative label. The same event can have different financial, operational, valuation, or financing effects.

---

## Engine 9: Historical Conditional Prediction Engine

### Core question
When the market historically looked like the current combined state, what happened next?

### Inputs

Outputs/states from Engines 1-8.

### Main calculations

Find historical observations with comparable states and calculate conditional distributions for multiple horizons.

Suggested initial horizons:

- 1D
- 2D
- 5D
- 10D
- 20D

### Output

- `historical_sample_size`
- `expected_return`
- `median_return`
- `positive_outcome_rate`
- `negative_outcome_rate`
- `return_distribution`
- `historical_percentile`

### Core principle

Do not simply predict BUY/SELL. Estimate what historically happened after comparable states.

---

## Engine 10: Model Validation + Auditor Engine

### Core question
Did the models actually work?

### Main calculations

For every engine and horizon compare prediction with realized outcome.

Track:

- directional accuracy
- MAE
- RMSE
- hit rate
- precision
- false-positive rate
- performance by regime
- performance by sector
- performance by volatility regime
- rolling performance
- calibration

### Output

Reliability statistics used by the Conviction Engine.

### Notes

Time alignment / information-availability controls are a required infrastructure layer for every engine, not merely another prediction engine.

---

# Conviction Engine

The Conviction Engine sits above Engines 1-10.

It should not create an independent signal. Its job is to evaluate the quality of the evidence generated by the other engines.

### Inputs

- engine outputs
- historical engine reliability
- current regime
- signal agreement
- signal disagreement
- sample size
- data quality
- horizon

### Main concepts

- direction
- strength
- agreement
- uncertainty
- reliability
- regime compatibility

Avoid a naive average such as:

```text
Engine 1 score + Engine 2 score + Engine 3 score ...
```

because many variables and engines may represent the same underlying risk regime. This can cause double-counting.

---

# Critical Statistical / Modelling Risks

## 1. Correlation is not causation

A historical correlation between USD_INR, Brent, VIX, FII flow, and Nifty does not prove a causal relationship.

Use:

- conditional relationships
- regime-specific relationships
- rolling relationships
- historical outcome distributions

instead of claiming deterministic causality.

## 2. Multicollinearity

Macro variables such as DXY, USD_INR, US10Y, Brent, VIX, and equity indices can move together.

Do not throw all variables into one giant regression.

## 3. Double counting

Several engines may capture the same latent regime.

Example:

```text
India VIX ↑
FII selling ↑
Nifty ↓
DXY ↑
USD_INR ↑
```

These should not automatically count as five independent pieces of evidence.

## 4. Regime changes

Historical relationships can change across:

- bull markets
- bear markets
- high-volatility regimes
- low-volatility regimes
- inflation regimes
- rate-hike regimes
- rate-cut regimes
- structural market changes

Use rolling and conditional analysis.

## 5. Look-ahead bias

Every feature must be based on information actually available at the prediction timestamp.

Preferred conceptual fields:

```text
source
published_at
market_timestamp
effective_date
timezone
```

Do not rely only on a simplistic `t-1` rule.

## 6. In-sample overconfidence

R² or in-sample model fit is not prediction confidence.

Use walk-forward / out-of-sample validation and calibration.

## 7. Frequency mismatch

Market data may be minute/5-minute/daily while fundamentals are quarterly/yearly.

Treat fundamentals as slower state variables rather than pretending all variables have the same temporal resolution.

## 8. Arbitrary thresholds

Avoid hard-coded values unless supported by historical testing.

Examples of rules requiring validation:

- VIX > X
- volume > 300%
- delivery > 60%
- predicted return / 5%
- fixed P/E standard-deviation bands

---

# Current OLS Engine: Known Issues

The current `olsEngine1.py` is a useful starting point but should not be treated as the finished predictive model.

Known concerns:

1. `confidence = R²` is not genuine prediction confidence.
2. No proper walk-forward validation is implemented in the core model.
3. The score normalisation uses a fixed 5% divisor that needs empirical justification.
4. `target_2d` uses future prices, so feature extraction must be carefully separated from the latest available observation.
5. The feature set is relatively narrow and should be tested for stability rather than assumed to be predictive.

Treat it as the first market-microstructure model, not as the final conviction engine.

---

# Recommended Data Flow

```text
RAW DATA
   │
   ▼
DATA QUALITY + NORMALIZATION
   │
   ▼
TIME / INFORMATION AVAILABILITY ALIGNMENT
   │
   ▼
FEATURE ENGINEERING
   │
   ├── Market Microstructure
   ├── Derivatives Positioning
   ├── Institutional Flow
   ├── Global + Macro Regime
   ├── Commodity Transmission
   ├── Fundamental Regime
   ├── Valuation + Rates
   └── News + Operational Events
   │
   ▼
REGIME / STATE REPRESENTATION
   │
   ▼
HISTORICAL CONDITIONAL PREDICTION
   │
   ▼
CONVICTION ENGINE
   │
   ▼
PREDICTION_LEDGER
   │
   ▼
REALIZED OUTCOME
   │
   ▼
AUDITOR / MODEL VALIDATION
```

---

# Execution Engine Is Separate

Intraday execution should not be part of the core prediction/conviction engine.

Separate:

```text
prediction
risk management
execution
```

An eventual execution layer may use:

- VWAP
- ATR
- intraday volume
- ticker-specific OI levels
- market structure

but it should consume the upstream prediction/conviction output rather than redefine it.

---

# Design Principle

The system should not be:

```text
10 models shouting BUY/SELL
```

It should be:

```text
multiple evidence engines
        ↓
current market/business state
        ↓
historical conditional outcome
        ↓
calibrated conviction
        ↓
validated prediction
```

The system's purpose is a **soft, evidence-based trend assessment**, not a claim of deterministic future prices.
