# News Analysis + Bell-Curve Context Plan

## 1. Objective

Build a financial-news analysis system that does not simply label a news article as bullish or bearish.

The system should:

1. extract the factual event from the news;
2. identify what type of information the event contains;
3. fetch the appropriate company, industry, financial, and market context;
4. measure the size/significance of the event relative to relevant historical and peer baselines;
5. produce an evidence-based significance profile;
6. later connect the event to the broader trend/conviction system.

The core idea is **contextual significance**, not an LLM-generated opinion.

---

## 2. Important Data-Domain Separation

The existing financial ETL is only one data domain.

### News data
Examples:
- investment announcement
- capacity expansion
- order win
- acquisition
- plant shutdown
- regulatory action
- management guidance
- project commissioning

### Financial data
Already available from `company_financial_data_ETL`:
- revenue
- profit
- assets
- debt
- equity
- cash flow
- capex
- valuation
- shareholding
- other financial-statement-derived metrics

### Company operational data
A separate dataset/source family:
- installed capacity
- production
- utilisation
- order book
- order inflow
- project pipeline
- plant information
- current/planned capacity
- commissioning status

### Industry operational data
A separate dataset/source family:
- industry capacity
- industry production
- demand
- industry growth
- competitor capacity
- market share
- industry utilisation

### Market / macro context
Already available in the existing ETL:
- Nifty 50
- India VIX
- US VIX
- USD/INR
- DXY
- Brent
- India 10Y
- US 10Y
- yield spread
- CPI
- broad commodities
- global assets
- F&O
- institutional flows
- trade events

---

## 3. News Analysis Pipeline

```text
NEWS ARTICLE
    |
    v
NEWS SCRAPER
    |
    v
NEWS PARSER / EVENT EXTRACTOR
    |
    v
STRUCTURED EVENT
    |
    +-------------------+-------------------+-------------------+
    |                   |                   |                   |
    v                   v                   v                   v
FINANCIAL CONTEXT   COMPANY OPS        INDUSTRY OPS       MARKET/MACRO
(existing ETL)      (new source)       (new source)       (existing ETL)
    |                   |                   |                   |
    +-------------------+-------------------+-------------------+
                            |
                            v
                    EVENT SIGNIFICANCE ENGINE
                            |
                            v
                     HISTORICAL BASELINE
                            |
                            v
                    PEER / INDUSTRY BASELINE
                            |
                            v
                     BELL-CURVE PROFILE
                            |
                            v
                    NEWS ANALYSIS OUTPUT
```

---

## 4. The News Parser

The parser should convert an article into structured facts rather than asking the LLM to directly decide the stock impact.

Example:

```json
{
  "company": "Solex Energy",
  "event_type": "capacity_expansion",
  "announcement_date": "YYYY-MM-DD",
  "investment_amount": 4000,
  "investment_currency": "INR_CRORE",
  "new_capacity": 2.2,
  "capacity_unit": "GW",
  "future_capacity": 5.2,
  "future_capacity_unit": "GW",
  "bess_capacity": 5,
  "bess_unit": "GWh",
  "commissioning_target": "FY28/FY29",
  "source": "...",
  "source_timestamp": "..."
}
```

The parser extracts facts. It should not invent the significance.

---

## 5. Event-Type Routing

The event type determines what context is required.

Examples:

### Capacity expansion
Needs:
- existing company capacity
- new capacity
- industry capacity
- investment
- historical capex
- revenue/assets/cash flow
- project timeline

### Order win
Needs:
- order value
- existing order book
- annual revenue
- order-book-to-revenue ratio
- sector/industry demand
- execution history

### Acquisition
Needs:
- acquisition value
- target size
- company cash
- debt
- valuation
- previous acquisitions
- expected contribution

### Plant shutdown
Needs:
- affected capacity
- total company capacity
- production
- utilisation
- revenue exposure
- industry supply context

The routing layer should be configuration-driven, not hard-coded inside an LLM prompt.

---

## 6. Bell-Curve Concept

The bell curve is intended to answer:

> How unusual or significant is this event compared with the company's normal history, peers, or industry?

It should NOT be one generic bullish/bearish curve.

Instead, measure separate dimensions.

### Example dimensions

```text
financial_magnitude
operational_magnitude
industry_magnitude
historical_unusualness
valuation_relevance
balance_sheet_impact
cash_flow_impact
execution_dependence
market_relevance
```

For each dimension, calculate an empirical position such as:
- percentile
- z-score
- deviation from historical mean/median
- peer-relative percentile

Example:

```text
Investment vs annual revenue       94th percentile
Investment vs historical capex     97th percentile
New capacity vs existing capacity  91st percentile
New capacity vs industry capacity  78th percentile
```

This gives meaning to a statement like:

> "₹4,000 crore investment"

without assuming that ₹4,000 crore is inherently large.

---

## 7. Why One Overall Bell Curve Is Insufficient

An event can be:

- very large operationally,
- financially demanding,
- positive for long-term capacity,
- but dependent on execution,
- and potentially dilutive or debt-intensive.

Therefore avoid:

```text
EVENT SCORE = +0.82
```

as the primary analytical output.

Prefer:

```text
Operational significance      High
Financial magnitude           Very High
Industry significance         Moderate
Balance-sheet pressure        Moderate
Execution dependency          High
Historical unusualness        Very High
```

A later combined score can be introduced only after the component measures are empirically validated.

---

## 8. Historical Conditioning

The bell curve should eventually become more than descriptive.

For similar historical events, measure what happened afterward.

Example:

```text
similar capacity expansions
        |
        +--> next 1D return
        +--> next 5D return
        +--> next 20D return
        +--> next earnings impact
```

This separates:

```text
EVENT MAGNITUDE
```

from:

```text
HISTORICAL MARKET RESPONSE
```

Those are not the same thing.

---

## 9. LLM Role

The LLM should mainly handle unstructured information:

### Good uses
- identify event type
- extract facts
- normalize terminology
- identify affected company/industry
- identify missing fields
- summarize evidence
- explain calculated results

### Bad uses
- invent financial impact
- decide that an event is "very bullish"
- manufacture peer capacity
- estimate exact margins without data
- infer causality from one article
- replace statistical validation

The calculations should be executed by Python/DuckDB/Polars and deterministic configuration/rules.

---

## 10. Planned New Scrapers

The news system introduces four new data pipelines around the existing financial ETL:

```text
1. NEWS SCRAPER
2. COMPANY OPERATIONAL DATA SCRAPER
3. INDUSTRY OPERATIONAL DATA SCRAPER
4. GOVERNMENT / SECTOR DATA SCRAPER
```

The existing financial ETL remains the financial-data provider.

---

## 11. Initial Engine Plan

The news-analysis side should start smaller than the trend-prediction system.

### Engine A: News Event Extraction
Purpose:
- turn article text into structured event facts.

### Engine B: Context Resolver
Purpose:
- determine which datasets are needed for the event.

### Engine C: Event Significance Engine
Purpose:
- calculate event magnitude against company/industry/history.

### Engine D: Historical Event Baseline
Purpose:
- find comparable historical events and measure subsequent outcomes.

### Engine E: News Interpretation Layer
Purpose:
- explain the quantitative evidence in plain language.

Later:

### Engine F: News-to-Market Interaction
Connects significant events to the existing market/macro/fundamental trend system.

---

## 12. Example: Solex Capacity Announcement

News:

```text
investment = ₹4,000 crore
new cell capacity = 2.2 GW
future capacity = 5.2 GW
BESS = 5 GWh
```

The system should NOT immediately say:

```text
bullish
```

It should retrieve:

```text
company financial:
    revenue
    assets
    cash flow
    historical capex

company operational:
    existing cell capacity
    existing manufacturing footprint
    utilisation
    production

industry operational:
    total cell capacity
    competitor capacity
    demand
    industry growth
```

Then calculate:

```text
₹4,000 crore / company revenue
₹4,000 crore / total assets
₹4,000 crore / historical capex
2.2 GW / existing company capacity
2.2 GW / industry capacity
5.2 GW / industry capacity
```

The bell-curve layer then tells us how unusual those numbers are relative to defined baselines.

---

## 13. Data Quality Requirement

Every news analysis must retain:

```text
source
publication_timestamp
event_timestamp
data_as_of
source_type
confidence_in_extraction
missing_context_fields
```

The system must distinguish:

```text
known fact
estimated value
derived calculation
historical comparison
LLM interpretation
```

This is especially important because news articles frequently omit definitions, units, timelines, or whether a number is current versus planned.

---

## 14. Development Order

Do not build all scrapers and engines simultaneously.

Recommended sequence:

```text
PHASE 1
Define event schema
        |
        v
PHASE 2
Build news scraper + extraction
        |
        v
PHASE 3
Define operational data schema
        |
        v
PHASE 4
Build company operational source
        |
        v
PHASE 5
Build industry / government sources
        |
        v
PHASE 6
Build event significance calculations
        |
        v
PHASE 7
Build bell-curve / percentile baseline
        |
        v
PHASE 8
Historical event study
        |
        v
PHASE 9
Connect to trend / conviction system
```

---

## 15. Core Design Principle

```text
NEWS DOES NOT PRODUCE THE VERDICT.

NEWS PRODUCES AN EVENT.

THE EVENT DETERMINES REQUIRED CONTEXT.

CONTEXT PRODUCES MEASUREMENTS.

MEASUREMENTS ARE COMPARED WITH HISTORY / PEERS.

THAT PRODUCES EVENT SIGNIFICANCE.

ONLY AFTER VALIDATION SHOULD IT AFFECT
THE BROADER MARKET CONVICTION SYSTEM.
```

This is the working context for the **News Analysis + Bell-Curve** discussion. It should remain separate from the current Engine 1 market-prediction work until the interfaces between the two systems are explicitly defined.
