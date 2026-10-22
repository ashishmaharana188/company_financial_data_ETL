# Commodity derivatives in TrendAnalysis

The app reads commodity contracts directly from `unified_market_master`. It does not use `mv_spot_futures_basis` for MCX contracts because that legacy table joins futures to NSE cash-equity rows.

## Market-price selection

The **Market prices → Price universe → MCX commodities (daily OHLCV)** option loads commodity futures from `unified_market_master` using the `FUTCOM` + non-option-side filter. The chart can plot Open, High, Low, Close or Volume, and the terminal readout retains the raw OHLC values. It uses the highest-open-interest futures contract per date (earlier expiry breaks ties), so the represented expiry can roll through time. This is a stitched daily series, not an expiry-fixed contract history.

The market-price source is distinct from global/equity assets because those use `global_assets_daily`; the MCX history is sourced from raw derivative rows. This option does not manufacture data if the required `FUTCOM` rows are absent. The current query is daily (`ReportDate`); intraday OHLC requires intraday bars to be ingested and exposed separately.

## Instrument classification

- `FUTCOM` with `OptionType` `XX` or blank: commodity futures.
- `OPTFUT` or `OPTCOM` with `OptionType` `CE` or `PE`: commodity options.
- Stock/index futures and options stay in separate pickers.

Both the identifier catalogue and the chart-history query use the same predicates. This is deliberate: a contract should not appear in the picker unless the series loader uses the same instrument classification.

## In-app coverage diagnostic

Open **Options → Commodity options** or **Futures → Commodity futures contracts** and expand **MCX / commodity raw-data coverage**. It groups the raw table by instrument type and option side and shows row count, unique tickers and date coverage. If no supported rows are found, the app reports that before any chart query is attempted.

If records exist but do not appear in the selected picker, inspect the `Option side` column and the source instrument code. Do not silently remap commodity names to macro tickers: for example, a Brent benchmark and an MCX crude-oil future may be related instruments without being identical series.

## Direct DuckDB inventory check

```sql
SELECT
    UPPER(TRIM(CAST("InstrumentType" AS VARCHAR))) AS instrument_type,
    UPPER(TRIM(COALESCE(CAST("OptionType" AS VARCHAR), 'XX'))) AS option_type,
    COUNT(*) AS row_count,
    COUNT(DISTINCT "Ticker") AS ticker_count,
    MIN("ReportDate") AS first_date,
    MAX("ReportDate") AS latest_date
FROM unified_market_master
WHERE UPPER(TRIM(COALESCE(CAST("InstrumentType" AS VARCHAR), ''))) IN ('FUTCOM', 'OPTFUT', 'OPTCOM')
GROUP BY 1, 2
ORDER BY 1, 2;
```

No local production DuckDB is bundled with this overlay. The coverage panel and query above should be used to verify the actual installed dataset.
