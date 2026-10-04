import argparse
import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import duckdb
import numpy as np
import polars as pl

from scripts.database import DB_PATH


class OLSMicrostructureEngine:
    """
    Engine 1: Market Microstructure.

    Scope is deliberately limited to the market microstructure matrix:
    price/returns, volume, delivery, options PCR, futures basis, trading-range
    behaviour, VWAP deviation, block-event footprint and short-volume footprint.

    Macro, FII/DII, commodities, fundamentals and valuation do not enter this
    engine. Those belong to later engines and the eventual conviction layer.
    """

    ENGINE_NAME = "1_OLS_Microstructure"

    BASE_COLUMNS = [
        "ticker",
        "date",
        "close",
        "volume",
        "delivery_percentage",
        "daily_hl_spread",
        "daily_vwap_dev",
        "oi_pcr",
        "delta_oi_pcr",
        "futures_basis",
        "net_block_volume",
        "avg_block_premium",
        "short_percentage",
    ]

    FEATURE_COLUMNS = [
        "return_1d",
        "return_3d",
        "return_5d",
        "volume_ratio_20d",
        "volume_zscore_20d",
        "delivery_zscore_20d",
        "pcr_change_3d",
        "pcr_zscore_20d",
        "basis_change_3d",
        "basis_zscore_20d",
        "hl_spread_zscore_20d",
        "vwap_dev_zscore_20d",
        "block_volume_ratio_20d",
        "block_premium_zscore_20d",
        "short_percentage_zscore_20d",
    ]

    HORIZONS = {"2D": 2, "5D": 5}
    MIN_TRAIN_ROWS = 80
    DIAGNOSTIC_MIN_TRAIN_ROWS = 120

    def __init__(self, engine_name: str = ENGINE_NAME):
        self.engine_name = engine_name

    # ------------------------------------------------------------------
    # Native DuckDB / Polars access
    # ------------------------------------------------------------------
    @staticmethod
    def _read_polars(query: str, params: Optional[dict] = None) -> pl.DataFrame:
        """Execute an OLAP query directly and return a Polars DataFrame."""
        with duckdb.connect(DB_PATH, read_only=True) as con:
            if params:
                result = con.execute(query, params)
            else:
                result = con.execute(query)
            try:
                return result.pl()
            except AttributeError:
                return pl.from_arrow(result.fetch_arrow_table())

    @staticmethod
    def _read_scalar(query: str, params: Optional[dict] = None):
        with duckdb.connect(DB_PATH, read_only=True) as con:
            row = con.execute(query, params or {}).fetchone()
            return row[0] if row else None

    @staticmethod
    def _ensure_prediction_ledger(con: duckdb.DuckDBPyConnection) -> None:
        """Keep Engine 1 self-sufficient without changing database.py."""
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_ledger (
                "engine_name" VARCHAR,
                "ticker" VARCHAR,
                "asof_date" DATE,
                "horizon" VARCHAR,
                "signal" VARCHAR,
                "score" DOUBLE,
                "confidence" DOUBLE,
                "veto_flag" BOOLEAN DEFAULT false,
                "penalty" DOUBLE DEFAULT 0.0,
                "target_metric" VARCHAR,
                "reason_json" JSON,
                "feature_json" JSON,
                "data_quality_score" DOUBLE,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY ("engine_name", "ticker", "asof_date", "horizon")
            );
            """
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def fetch_historical_matrix(
        self, ticker: str, asof_date: str, lookback: int = 750
    ) -> pl.DataFrame:
        query = """
            SELECT
                ticker,
                date,
                close,
                volume,
                delivery_percentage,
                daily_hl_spread,
                daily_vwap_dev,
                oi_pcr,
                delta_oi_pcr,
                futures_basis,
                net_block_volume,
                avg_block_premium,
                short_percentage
            FROM mv_unified_market_matrix
            WHERE ticker = $ticker
              AND date <= CAST($asof_date AS DATE)
            ORDER BY date ASC
            LIMIT $limit
        """

        df = self._read_polars(
            query,
            {"ticker": ticker, "asof_date": asof_date, "limit": int(lookback)},
        )

        if df.is_empty():
            return df

        numeric = [c for c in self.BASE_COLUMNS if c not in {"ticker", "date"}]
        df = df.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric]
        )
        return (
            df.with_columns(pl.col("date").cast(pl.Date, strict=False))
            .drop_nulls("date")
            .sort("date")
            .unique(subset=["date"], keep="last")
            .sort("date")
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def _zscore_expression(column: str, window: int = 20) -> pl.Expr:
        mean_expr = pl.col(column).rolling_mean(window_size=window)
        std_expr = pl.col(column).rolling_std(window_size=window)
        return (
            pl.when(std_expr.is_not_null() & (std_expr > 0))
            .then((pl.col(column) - mean_expr) / std_expr)
            .otherwise(None)
        )

    def build_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        optional_defaults = {
            "volume": 0.0,
            "delivery_percentage": None,
            "daily_hl_spread": None,
            "daily_vwap_dev": None,
            "oi_pcr": None,
            "delta_oi_pcr": None,
            "futures_basis": None,
            "net_block_volume": 0.0,
            "avg_block_premium": None,
            "short_percentage": None,
        }

        additions = []
        for column, default in optional_defaults.items():
            if column not in df.columns:
                additions.append(pl.lit(default).cast(pl.Float64).alias(column))
        if additions:
            df = df.with_columns(additions)

        df = df.with_columns(
            [
                pl.col("close").pct_change().alias("return_1d"),
                pl.col("close").pct_change(3).alias("return_3d"),
                pl.col("close").pct_change(5).alias("return_5d"),
            ]
        )

        volume_mean = pl.col("volume").rolling_mean(window_size=20)
        volume_std = pl.col("volume").rolling_std(window_size=20)
        df = df.with_columns(
            [
                pl.when(volume_mean.is_not_null() & (volume_mean > 0))
                .then(pl.col("volume") / volume_mean)
                .otherwise(None)
                .alias("volume_ratio_20d"),
                pl.when(volume_std.is_not_null() & (volume_std > 0))
                .then((pl.col("volume") - volume_mean) / volume_std)
                .otherwise(None)
                .alias("volume_zscore_20d"),
            ]
        )

        df = df.with_columns(
            [
                self._zscore_expression("delivery_percentage").alias(
                    "delivery_zscore_20d"
                ),
                pl.col("oi_pcr").diff(3).alias("pcr_change_3d"),
                self._zscore_expression("oi_pcr").alias("pcr_zscore_20d"),
                pl.col("futures_basis").diff(3).alias("basis_change_3d"),
                self._zscore_expression("futures_basis").alias("basis_zscore_20d"),
                self._zscore_expression("daily_hl_spread").alias(
                    "hl_spread_zscore_20d"
                ),
                self._zscore_expression("daily_vwap_dev").alias("vwap_dev_zscore_20d"),
            ]
        )

        df = df.with_columns(
            pl.col("net_block_volume").fill_null(0.0).abs().alias("abs_block_volume")
        )
        block_mean = pl.col("abs_block_volume").rolling_mean(window_size=20)
        df = df.with_columns(
            pl.when(block_mean.is_not_null() & (block_mean > 0))
            .then(pl.col("abs_block_volume") / block_mean)
            .otherwise(None)
            .alias("block_volume_ratio_20d")
        )

        df = df.with_columns(
            [
                self._zscore_expression("avg_block_premium").alias(
                    "block_premium_zscore_20d"
                ),
                self._zscore_expression("short_percentage").alias(
                    "short_percentage_zscore_20d"
                ),
            ]
        )

        return df.drop("abs_block_volume")

    def add_forward_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                (pl.col("close").shift(-days) / pl.col("close") - 1.0).alias(
                    f"target_{label.lower()}"
                )
                for label, days in self.HORIZONS.items()
            ]
        )

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def _prepare_training_frame(self, df: pl.DataFrame, horizon: str) -> pl.DataFrame:
        target = f"target_{horizon.lower()}"
        selected = df.select(self.FEATURE_COLUMNS + [target]).drop_nulls()

        if selected.is_empty():
            return selected

        arrays = [
            selected.select(self.FEATURE_COLUMNS).to_numpy(),
            selected[target].to_numpy(),
        ]
        x, y = arrays
        mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if not mask.any():
            return pl.DataFrame(schema=selected.schema)

        return selected.filter(pl.Series("valid", mask))

    @staticmethod
    def _safe_float(value, default=None):
        try:
            number = float(value)
            return number if np.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    def fit_ols(self, training_df: pl.DataFrame, horizon: str) -> Optional[Dict]:
        target = f"target_{horizon.lower()}"
        if training_df.height < self.MIN_TRAIN_ROWS:
            return None

        x_all = training_df.select(self.FEATURE_COLUMNS).to_numpy().astype(float)
        y_all = training_df.select(target).to_numpy().ravel().astype(float)

        usable = []
        for index, feature in enumerate(self.FEATURE_COLUMNS):
            column = x_all[:, index]
            if (
                np.isfinite(column).all()
                and np.std(column) > 1e-12
                and np.unique(column).size > 1
            ):
                usable.append(feature)

        if len(usable) < 3:
            return None

        indices = [self.FEATURE_COLUMNS.index(feature) for feature in usable]
        x = x_all[:, indices]
        x_aug = np.column_stack([np.ones(x.shape[0]), x])

        try:
            beta, _, rank, singular_values = np.linalg.lstsq(x_aug, y_all, rcond=None)
            fitted_values = x_aug @ beta
        except np.linalg.LinAlgError:
            return None

        residuals = y_all - fitted_values
        sse = float(np.sum(residuals**2))
        sst = float(np.sum((y_all - np.mean(y_all)) ** 2))
        r_squared = 1.0 - sse / sst if sst > 0 else 0.0
        n = len(y_all)
        p = len(usable)
        adjusted_r2 = (
            1.0 - (1.0 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        )

        residual_std = float(np.std(residuals, ddof=1)) if n > 1 else np.nan
        target_std = float(np.std(y_all, ddof=1)) if n > 1 else np.nan

        return {
            "beta": beta,
            "usable_features": usable,
            "r_squared": self._safe_float(r_squared, 0.0),
            "adjusted_r_squared": self._safe_float(adjusted_r2, 0.0),
            "residual_std": residual_std,
            "target_std": target_std,
            "training_rows": n,
            "rank": int(rank),
            "condition_number": self._safe_float(np.linalg.cond(x_aug), None),
            "target_mean": self._safe_float(np.mean(y_all), 0.0),
            "target_median": self._safe_float(np.median(y_all), 0.0),
            "target_p05": self._safe_float(np.quantile(y_all, 0.05), 0.0),
            "target_p95": self._safe_float(np.quantile(y_all, 0.95), 0.0),
        }

    def predict_current(self, fitted: Dict, current_row: Dict) -> Optional[Dict]:
        values = []
        for feature in fitted["usable_features"]:
            value = self._safe_float(current_row.get(feature), None)
            if value is None:
                return None
            values.append(value)

        x_aug = np.asarray([1.0] + values, dtype=float)
        predicted_return = self._safe_float(float(x_aug @ fitted["beta"]), 0.0)

        residual_std = fitted.get("residual_std")
        if residual_std is not None and np.isfinite(residual_std) and residual_std > 0:
            lower = predicted_return - 1.96 * residual_std
            upper = predicted_return + 1.96 * residual_std
        else:
            lower = None
            upper = None

        target_std = fitted.get("target_std")
        if target_std is not None and np.isfinite(target_std) and target_std > 0:
            standardized = predicted_return / target_std
        else:
            standardized = 0.0

        score = float(np.tanh(standardized))
        direction = "NEUTRAL"
        if score > 0.15:
            direction = "BULLISH"
        elif score < -0.15:
            direction = "BEARISH"

        return {
            "expected_return": predicted_return,
            "prediction_lower": lower,
            "prediction_upper": upper,
            "direction_score": score,
            "direction": direction,
        }

    # ------------------------------------------------------------------
    # Walk-forward diagnostics
    # ------------------------------------------------------------------
    def walk_forward_diagnostics(
        self,
        df: pl.DataFrame,
        horizon: str,
        min_train_rows: int = DIAGNOSTIC_MIN_TRAIN_ROWS,
        step: int = 5,
    ) -> Dict:
        target = f"target_{horizon.lower()}"
        working = self._prepare_training_frame(df, horizon)
        if working.height < min_train_rows + 1:
            return {
                "sample_size": 0,
                "directional_accuracy": None,
                "mae": None,
                "rmse": None,
            }

        predictions: List[float] = []
        actuals: List[float] = []

        for test_position in range(min_train_rows, working.height, step):
            train_slice = working.slice(0, test_position)
            test_slice = working.slice(test_position, step)
            if test_slice.is_empty():
                continue

            fitted = self.fit_ols(train_slice, horizon)
            if not fitted:
                continue

            for row in test_slice.iter_rows(named=True):
                prediction = self.predict_current(fitted, row)
                if prediction is None:
                    continue
                predictions.append(prediction["expected_return"])
                actuals.append(self._safe_float(row[target], 0.0))

        if not predictions:
            return {
                "sample_size": 0,
                "directional_accuracy": None,
                "mae": None,
                "rmse": None,
            }

        pred = np.asarray(predictions, dtype=float)
        act = np.asarray(actuals, dtype=float)

        return {
            "sample_size": int(len(pred)),
            "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(act))),
            "mae": float(np.mean(np.abs(act - pred))),
            "rmse": float(np.sqrt(np.mean((act - pred) ** 2))),
        }

    # ------------------------------------------------------------------
    # Prediction records + storage
    # ------------------------------------------------------------------
    def calculate_data_quality(self, current_row: Dict) -> float:
        required = [
            "close",
            "volume",
            "return_1d",
            "volume_ratio_20d",
            "delivery_zscore_20d",
            "pcr_zscore_20d",
            "basis_zscore_20d",
        ]
        available = sum(
            self._safe_float(current_row.get(column), None) is not None
            for column in required
        )
        return float(available / len(required))

    def build_prediction_record(
        self,
        ticker: str,
        asof_date: str,
        horizon: str,
        fitted: Dict,
        prediction: Dict,
        current_row: Dict,
        diagnostics: Dict,
    ) -> Dict:
        feature_snapshot = {
            key: self._safe_float(current_row.get(key), None)
            for key in self.FEATURE_COLUMNS
        }
        for key in [
            "close",
            "volume",
            "delivery_percentage",
            "daily_hl_spread",
            "daily_vwap_dev",
            "oi_pcr",
            "delta_oi_pcr",
            "futures_basis",
            "net_block_volume",
            "avg_block_premium",
            "short_percentage",
        ]:
            feature_snapshot[key] = self._safe_float(current_row.get(key), None)
        feature_snapshot["data_date"] = str(current_row.get("date"))

        reason = {
            "engine": self.engine_name,
            "scope": "market_microstructure_only",
            "horizon": horizon,
            "model": "OLS",
            "direction": prediction["direction"],
            "expected_return": prediction["expected_return"],
            "prediction_interval_95": {
                "lower": prediction["prediction_lower"],
                "upper": prediction["prediction_upper"],
            },
            "model_fit": {
                "r_squared": fitted["r_squared"],
                "adjusted_r_squared": fitted["adjusted_r_squared"],
                "training_rows": fitted["training_rows"],
                "rank": fitted["rank"],
                "condition_number": fitted["condition_number"],
            },
            "walk_forward": diagnostics,
            "target_distribution": {
                "mean": fitted["target_mean"],
                "median": fitted["target_median"],
                "p05": fitted["target_p05"],
                "p95": fitted["target_p95"],
            },
            "note": (
                "No macro, institutional-flow, fundamental, commodity, valuation, "
                "penalty or veto logic is applied inside Engine 1."
            ),
        }

        return {
            "engine_name": self.engine_name,
            "ticker": ticker,
            "asof_date": asof_date,
            "horizon": horizon,
            "signal": prediction["direction"],
            "score": prediction["direction_score"],
            "confidence": diagnostics.get("directional_accuracy"),
            "veto_flag": False,
            "penalty": 0.0,
            "target_metric": json.dumps(
                {
                    "expected_return": prediction["expected_return"],
                    "prediction_lower": prediction["prediction_lower"],
                    "prediction_upper": prediction["prediction_upper"],
                }
            ),
            "reason_json": json.dumps(reason, allow_nan=False),
            "feature_json": json.dumps(feature_snapshot, allow_nan=False),
            "data_quality_score": self.calculate_data_quality(current_row),
        }

    def store_predictions(self, records: List[Dict]) -> None:
        if not records:
            return

        with duckdb.connect(DB_PATH, read_only=False) as con:
            self._ensure_prediction_ledger(con)

            sql = """
                INSERT INTO prediction_ledger (
                    "engine_name", "ticker", "asof_date", "horizon", "signal",
                    "score", "confidence", "veto_flag", "penalty", "target_metric",
                    "reason_json", "feature_json", "data_quality_score"
                )
                VALUES (?, ?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), CAST(? AS JSON), ?)
                ON CONFLICT ("engine_name", "ticker", "asof_date", "horizon")
                DO UPDATE SET
                    "signal" = EXCLUDED."signal",
                    "score" = EXCLUDED."score",
                    "confidence" = EXCLUDED."confidence",
                    "veto_flag" = EXCLUDED."veto_flag",
                    "penalty" = EXCLUDED."penalty",
                    "target_metric" = EXCLUDED."target_metric",
                    "reason_json" = EXCLUDED."reason_json",
                    "feature_json" = EXCLUDED."feature_json",
                    "data_quality_score" = EXCLUDED."data_quality_score",
                    "created_at" = CURRENT_TIMESTAMP
            """

            for record in records:
                con.execute(
                    sql,
                    [
                        record["engine_name"],
                        record["ticker"],
                        record["asof_date"],
                        record["horizon"],
                        record["signal"],
                        record["score"],
                        record["confidence"],
                        record["veto_flag"],
                        record["penalty"],
                        record["target_metric"],
                        record["reason_json"],
                        record["feature_json"],
                        record["data_quality_score"],
                    ],
                )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def execute_pipeline(
        self,
        ticker: str,
        asof_date: Optional[str] = None,
        lookback: int = 750,
        run_diagnostics: bool = True,
    ) -> Dict:
        if not asof_date:
            asof_date = datetime.now().strftime("%Y-%m-%d")

        raw = self.fetch_historical_matrix(ticker, asof_date, lookback)
        if raw.is_empty():
            return {"ticker": ticker, "status": "NO_DATA", "predictions": []}

        featured = self.add_forward_targets(self.build_features(raw))
        current_rows = featured.filter(
            pl.col("date") <= pl.lit(date.fromisoformat(asof_date))
        )
        if current_rows.is_empty():
            return {"ticker": ticker, "status": "NO_CURRENT_ROW", "predictions": []}

        current_row = current_rows.tail(1).to_dicts()[0]
        records: List[Dict] = []
        diagnostics_by_horizon: Dict[str, Dict] = {}

        for horizon in self.HORIZONS:
            training_df = self._prepare_training_frame(featured, horizon)
            fitted = self.fit_ols(training_df, horizon)
            if not fitted:
                diagnostics_by_horizon[horizon] = {
                    "sample_size": 0,
                    "reason": "insufficient_training_data",
                }
                continue

            prediction = self.predict_current(fitted, current_row)
            if prediction is None:
                diagnostics_by_horizon[horizon] = {
                    "sample_size": 0,
                    "reason": "incomplete_current_features",
                }
                continue

            diagnostics = (
                self.walk_forward_diagnostics(featured, horizon)
                if run_diagnostics
                else {
                    "sample_size": 0,
                    "directional_accuracy": None,
                    "mae": None,
                    "rmse": None,
                }
            )
            diagnostics_by_horizon[horizon] = diagnostics

            records.append(
                self.build_prediction_record(
                    ticker=ticker,
                    asof_date=str(current_row["date"]),
                    horizon=horizon,
                    fitted=fitted,
                    prediction=prediction,
                    current_row=current_row,
                    diagnostics=diagnostics,
                )
            )

        self.store_predictions(records)

        return {
            "ticker": ticker,
            "status": "SUCCESS" if records else "NO_PREDICTION",
            "asof_date": str(current_row["date"]),
            "records": records,
            "diagnostics": diagnostics_by_horizon,
        }

    def run_mass_historical_backfill(
        self,
        days_depth: int = 60,
        lookback: int = 750,
        ticker_limit: Optional[int] = None,
    ) -> None:
        ticker_df = self._read_polars(
            'SELECT "Ticker" FROM market_metadata WHERE "IsActive" = true ORDER BY "Ticker"'
        )
        tickers = ticker_df.get_column("Ticker").drop_nulls().cast(pl.String).to_list()
        if ticker_limit:
            tickers = tickers[:ticker_limit]

        max_date = self._read_scalar("SELECT MAX(date) FROM mv_unified_market_matrix")
        if not max_date:
            print("[ERROR] No dates available in mv_unified_market_matrix.")
            return

        end_date = (
            max_date
            if isinstance(max_date, date)
            else date.fromisoformat(str(max_date))
        )
        business_dates = []
        cursor = end_date
        while len(business_dates) < days_depth:
            if cursor.weekday() < 5:
                business_dates.append(cursor)
            cursor -= timedelta(days=1)

        print(
            f"[START] Engine 1 backfill: {len(tickers)} tickers x {len(business_dates)} business dates"
        )

        for target_date in business_dates:
            asof = target_date.isoformat()
            for ticker in tickers:
                try:
                    self.execute_pipeline(
                        ticker=ticker,
                        asof_date=asof,
                        lookback=lookback,
                        run_diagnostics=False,
                    )
                except Exception as exc:
                    print(f"[SKIP] {ticker} {asof}: {exc}")

        print("[SUCCESS] Engine 1 historical backfill complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engine 1: Market Microstructure")
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--asof", type=str)
    parser.add_argument("--lookback", type=int, default=750)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--ticker-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = OLSMicrostructureEngine()

    if args.backfill:
        pipeline.run_mass_historical_backfill(
            days_depth=args.days,
            lookback=args.lookback,
            ticker_limit=args.ticker_limit,
        )
        return

    if not args.ticker:
        raise SystemExit("--ticker is required unless --backfill is used")

    result = pipeline.execute_pipeline(
        ticker=args.ticker,
        asof_date=args.asof,
        lookback=args.lookback,
        run_diagnostics=not args.no_diagnostics,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
