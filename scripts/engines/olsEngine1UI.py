import json
from datetime import datetime

import pandas as pd
import streamlit as st

from scripts.database import engine
from scripts.engines.olsEngine1 import (
    OLSMicrostructureEngine,
)


ENGINE_NAME = "1_OLS_Microstructure"


def _parse_json(value):

    if isinstance(value, dict):
        return value

    if isinstance(value, str):

        try:
            return json.loads(value)

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return {}

    return {}


def fetch_latest_market_date(
    ticker: str,
):

    query = """
        SELECT MAX(date) AS latest_date
        FROM mv_unified_market_matrix
        WHERE ticker = $ticker
    """

    df = engine.execute(
        query,
        {
            "ticker": ticker,
        },
    ).pl()

    if df.is_empty():
        return None

    value = df.get_column("latest_date")[0]

    if value is None:
        return None

    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def fetch_data_readiness(
    ticker: str,
):

    query = """
        SELECT
            COUNT(*) AS rows_total,
            MIN(date) AS min_date,
            MAX(date) AS max_date,

            COUNT(*)
            FILTER (
                WHERE close IS NOT NULL
            ) AS close_rows,

            COUNT(*)
            FILTER (
                WHERE volume IS NOT NULL
            ) AS volume_rows,

            COUNT(*)
            FILTER (
                WHERE delivery_percentage IS NOT NULL
            ) AS delivery_rows,

            COUNT(*)
            FILTER (
                WHERE is_fo_eligible = 1
            ) AS fo_rows

        FROM mv_unified_market_matrix

        WHERE ticker = $ticker
    """

    df = engine.execute(
        query,
        {
            "ticker": ticker,
        },
    ).pl()

    if df.is_empty():

        return {
            "rows_total": 0,
            "min_date": None,
            "max_date": None,
            "close_rows": 0,
            "volume_rows": 0,
            "delivery_rows": 0,
            "fo_rows": 0,
        }

    return df.to_dicts()[0]


def fetch_market_history(
    ticker: str,
    target_date: str,
    days: int = 60,
):

    query = """
        SELECT
            date,
            close,
            volume,
            delivery_percentage,
            oi_pcr,
            futures_basis,
            net_block_volume,
            avg_block_premium,
            short_percentage
        FROM mv_unified_market_matrix
        WHERE ticker = $ticker
          AND date <= CAST($target_date AS DATE)
        ORDER BY date DESC
        LIMIT $limit
    """

    df = engine.execute(
        query,
        {
            "ticker": ticker,
            "target_date": target_date,
            "limit": days,
        },
    ).pl()

    if df.is_empty():
        return pd.DataFrame()

    return df.sort("date").to_pandas().set_index("date")


def fetch_prediction_ledger(
    ticker: str,
    target_date: str,
):

    query = """
        SELECT
            asof_date,
            horizon,
            signal,
            score,
            confidence,
            veto_flag,
            penalty,
            target_metric,
            reason_json,
            feature_json,
            data_quality_score
        FROM prediction_ledger
        WHERE ticker = $ticker
          AND asof_date = CAST($target_date AS DATE)
          AND engine_name = $engine_name
        ORDER BY
            CASE
                WHEN horizon = '2D' THEN 1
                WHEN horizon = '5D' THEN 2
                ELSE 3
            END
    """

    try:

        return (
            engine.execute(
                query,
                {
                    "ticker": ticker,
                    "target_date": target_date,
                    "engine_name": ENGINE_NAME,
                },
            )
            .pl()
            .to_pandas()
        )

    except Exception:
        return pd.DataFrame()


def render_prediction_card(
    record: pd.Series,
):

    reason = _parse_json(record.get("reason_json"))

    model_fit = reason.get(
        "model_fit",
        {},
    )

    walk_forward = reason.get(
        "walk_forward",
        {},
    )

    target_metric = _parse_json(record.get("target_metric"))

    expected_return = target_metric.get(
        "expected_return",
        0.0,
    )

    direction = record.get(
        "signal",
        "NEUTRAL",
    )

    score = record.get(
        "score",
        0.0,
    )

    confidence = record.get("confidence")

    quality = record.get(
        "data_quality_score",
        0.0,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Direction",
        direction,
    )

    c2.metric(
        "Expected Return",
        (
            f"{float(expected_return) * 100:.2f}%"
            if expected_return is not None
            else "N/A"
        ),
    )

    c3.metric(
        "Direction Score",
        f"{float(score):.3f}",
    )

    c4.metric(
        "Walk-Forward Accuracy",
        (f"{float(confidence) * 100:.1f}%" if confidence is not None else "N/A"),
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Training Rows",
        str(
            model_fit.get(
                "training_rows",
                "N/A",
            )
        ),
    )

    r2.metric(
        "R-Squared",
        (
            f"{float(model_fit.get('r_squared', 0.0)):.4f}"
            if model_fit.get("r_squared") is not None
            else "N/A"
        ),
    )

    r3.metric(
        "MAE",
        (
            f"{float(walk_forward.get('mae', 0.0)) * 100:.3f}%"
            if walk_forward.get("mae") is not None
            else "N/A"
        ),
    )

    r4.metric(
        "Data Quality",
        f"{float(quality) * 100:.1f}%",
    )

    st.caption(
        "R-Squared describes model fit. "
        "Walk-forward accuracy is used as the historical directional reliability measure."
    )

    with st.expander(
        "Current Feature Snapshot",
        expanded=False,
    ):

        features = _parse_json(record.get("feature_json"))

        feature_rows = [
            {
                "Feature": key,
                "Value": value,
            }
            for key, value in features.items()
        ]

        st.dataframe(
            pd.DataFrame(feature_rows),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander(
        "Model Diagnostics",
        expanded=False,
    ):

        st.json(reason)


def render_ols_engine_ui(
    selected_ticker: str,
):

    st.markdown("## Engine 1: Market Microstructure")

    st.caption(
        "Engine 1 consumes only mv_unified_market_matrix. "
        "Macro, institutional, fundamentals and news are not used here."
    )

    latest_market_date = fetch_latest_market_date(selected_ticker)

    if latest_market_date:

        default_date = datetime.strptime(
            latest_market_date,
            "%Y-%m-%d",
        ).date()

    else:

        default_date = datetime.now().date()

    target_date = st.date_input(
        "Analysis Date",
        value=default_date,
        key="engine1_analysis_date",
        help=("For the first run, leave this at the latest available market date."),
    )

    target_date_str = target_date.strftime("%Y-%m-%d")

    readiness = fetch_data_readiness(selected_ticker)

    st.markdown("### Data Readiness")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Rows",
        str(
            readiness.get(
                "rows_total",
                0,
            )
        ),
    )

    q2.metric(
        "History From",
        str(
            readiness.get(
                "min_date",
                "N/A",
            )
        ),
    )

    q3.metric(
        "History To",
        str(
            readiness.get(
                "max_date",
                "N/A",
            )
        ),
    )

    q4.metric(
        "F&O Rows",
        str(
            readiness.get(
                "fo_rows",
                0,
            )
        ),
    )

    run_col, note_col = st.columns([1, 4])

    with run_col:

        run_engine = st.button(
            "Run Engine 1",
            type="primary",
            key="run_engine1_button",
        )

    with note_col:

        st.caption(
            "Runs the model from the dashboard. No terminal command is required."
        )

    if run_engine:

        runner = OLSMicrostructureEngine()

        with st.spinner(f"Running Engine 1 for {selected_ticker}..."):

            try:

                result = runner.execute_pipeline(
                    ticker=selected_ticker,
                    asof_date=target_date_str,
                    lookback=750,
                    run_diagnostics=True,
                )

                st.session_state["engine1_last_result"] = result

            except Exception as exc:

                st.session_state["engine1_last_result"] = {
                    "status": "ERROR",
                    "ticker": selected_ticker,
                    "error": str(exc),
                    "predictions": [],
                }

    result = st.session_state.get("engine1_last_result")

    if result and result.get("ticker") == selected_ticker:

        status = result.get(
            "status",
            "UNKNOWN",
        )

        if status == "SUCCESS":

            st.success(f"Engine 1 completed for {selected_ticker}.")

            effective_date = result.get("effective_market_date")

            if effective_date:

                st.info(
                    f"Requested date: {target_date_str} | "
                    f"Effective market row: {effective_date}"
                )

        elif status == "ERROR":

            st.error(
                "Engine 1 failed: "
                + result.get(
                    "error",
                    "Unknown error",
                )
            )

        else:

            st.warning(f"Engine 1 status: {status}")

    prediction_df = fetch_prediction_ledger(
        selected_ticker,
        target_date_str,
    )

    if prediction_df.empty:

        st.info("No Engine 1 prediction is stored for this ticker/date yet.")

    else:

        st.markdown("### Engine 1 Prediction Output")

        for horizon in [
            "2D",
            "5D",
        ]:

            horizon_df = prediction_df[prediction_df["horizon"] == horizon]

            if horizon_df.empty:
                continue

            st.markdown(f"#### {horizon} Horizon")

            render_prediction_card(horizon_df.iloc[0])

    st.divider()

    history_df = fetch_market_history(
        selected_ticker,
        target_date_str,
        days=60,
    )

    if not history_df.empty:

        st.markdown("### Recent Market Microstructure")

        left, right = st.columns([3, 1])

        with left:

            st.line_chart(
                history_df[["close"]],
                height=320,
            )

        with right:

            st.bar_chart(
                history_df[["volume"]],
                height=320,
            )

        with st.expander(
            "Raw Market Matrix",
            expanded=False,
        ):

            st.dataframe(
                history_df.reset_index(),
                hide_index=True,
                use_container_width=True,
            )
