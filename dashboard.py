import streamlit as st
import pandas as pd
from scripts.database import engine
from scripts.statementScrape import run_etl_pipeline
from scripts.macroScrape import register_discovered_tickers
import subprocess
import sys
from datetime import datetime, timedelta
import duckdb
from scripts.engines.companyMetrics import render_company_metrics
from scripts.engines.olsEngine1 import OLSMicrostructureEngine

# Import fetchers for the Market Overview (since it still calculates cross-sectional data)
from scripts.ratioAnalysis import fetch_roic, fetch_fcf_yield

st.set_page_config(
    page_title="Swarm Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Swarm Intelligence")
st.sidebar.markdown("Institutional Quantitative Platform")
st.sidebar.divider()

app_mode = st.sidebar.radio(
    "NAVIGATION MENU",
    ["Data Center", "Company Data", "Engines", "Market Overview"],
    key="main_nav_radio",
)

# --- NEW: Contextual Engine Sidebar Dropdown ---
engine_view = None
if app_mode == "Engines":
    st.sidebar.divider()
    st.sidebar.markdown("### Engine View Options")
    engine_view = st.sidebar.selectbox(
        "Select Active Engine:",
        ["OLS Engine 1", "Canvas Mode (All Engines)"],
        help="Select a specific engine or render all engines simultaneously on the canvas.",
    )

st.sidebar.divider()
st.sidebar.markdown("### Global Settings")
selected_source = st.sidebar.selectbox(
    "Primary Data Source",
    options=["vantage", "yfinance", "screener"],
    index=0,
    help="Strictly isolates all mathematical models to data provided by this specific spigot.",
)

st.sidebar.divider()
st.sidebar.caption("System Status: Online")
st.sidebar.caption("Database: DuckDB Connected")


def run_orchestrator(mode, start=None, end=None):
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.downloadOrchestrator",
        "--mode",
        mode,
    ]
    if start and end:
        cmd.extend(["--start", start, "--end", end])

    # Changed from subprocess.run to Popen to allow real-time log streaming
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    return process


def execute_pipeline_live(mode, start_dt=None, end_dt=None):
    """
    Runs the orchestrator as a subprocess.
    Logs are pushed strictly to the terminal to prevent Streamlit DOM freezing.
    """
    import subprocess
    import sys

    cmd = [sys.executable, "-u", "-m", "scripts.downloadOrchestrator", "--mode", mode]

    if start_dt and end_dt:
        cmd.extend(["--start", start_dt, "--end", end_dt])

    # 1. UI Feedback: Tell the user to look at the terminal
    st.info(
        f" Pipeline '{mode}' initialized. Please check your terminal for live execution logs."
    )

    try:
        # 2. Run the process (Output naturally flows to the terminal where Streamlit was launched)
        process = subprocess.run(
            cmd, text=True, check=False  # We handle the return code manually below
        )

        # 3. UI Feedback: Final Status
        if process.returncode == 0:
            st.success(f"Pipeline '{mode}' executed successfully!")
        else:
            st.error(
                f"Pipeline '{mode}' failed with exit code {process.returncode}. Check terminal for errors."
            )

    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")


if app_mode == "Data Center":
    st.title("Institutional Data Center")
    st.markdown(
        "Manage the Extraction, Transformation, and Loading (ETL) of market data."
    )

    tab_delta, tab_scrape, tab_ingest = st.tabs(
        [
            "Delta Bridge (Daily Sync)",
            "Isolated Bulk Scraper",
            "Master DB Sync",
        ]
    )

    # --- TAB 1: DELTA BRIDGE ---
    with tab_delta:
        st.subheader("Delta Bridge Protocol")
        st.markdown(
            """
        The Delta Bridge automatically checks the database for the latest available date, applies a **2-day overlap safety lag**, and scrapes/parses all missing data up to today. 
        It finishes by refreshing the **Alpha Factory** (Materialized Views).
        """
        )

        if st.button("Trigger Delta Bridge Synchronizer", type="primary"):
            execute_pipeline_live(mode="delta")

    # --- TAB 2: ISOLATED BULK SCRAPER ---
    with tab_scrape:
        st.subheader("Mass Historical Extraction")
        st.markdown(
            "Downloads raw zip/csv files to the `offline_data_cache` without touching the database."
        )

        col1, col2 = st.columns(2)
        with col1:
            # Default to 2015 genesis
            scrape_start = st.date_input(
                "Start Date", value=pd.to_datetime("2015-01-01"), key="scrape_start"
            )
        with col2:
            scrape_end = st.date_input(
                "End Date", value=datetime.now(), key="scrape_end"
            )

        if st.button("Run Scrapers Only"):
            execute_pipeline_live(
                mode="scrape_only",
                start_dt=scrape_start.strftime("%Y-%m-%d"),
                end_dt=scrape_end.strftime("%Y-%m-%d"),
            )

    # --- TAB 3: ISOLATED BULK INGESTION ---
    with tab_ingest:
        st.subheader("Master DB Sync (Dumb Loaders & Alpha Factory)")
        st.markdown(
            """
        Bypasses scrapers and pushes all files currently sitting in the `offline_data_cache` into the **Unified Master**, **Macro**, and **Ledger** tables. 
        It finishes by rebuilding the **Alpha Factory**.
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            ingest_start = st.date_input(
                "Start Date", value=pd.to_datetime("2015-01-01"), key="ingest_start"
            )
        with col2:
            ingest_end = st.date_input(
                "End Date", value=datetime.now(), key="ingest_end"
            )

        if st.button("Run Master Parse Sync", type="primary", key="master_sync_btn"):
            execute_pipeline_live(
                mode="bulk_historic",
                start_dt=ingest_start.strftime("%Y-%m-%d"),
                end_dt=ingest_end.strftime("%Y-%m-%d"),
            )

        st.divider()
        st.subheader("Macro & Global Assets Sync")
        st.markdown(
            "Fetch real-time yields, commodities, and global indexes to update the Macro Ledgers."
        )

        # The clean, isolated 2-liner for Macro Refresh!
        if st.button(
            "Trigger Macro Refresh", type="secondary", key="macro_refresh_btn"
        ):
            execute_pipeline_live(mode="macro_refresh")

        if st.button(
            "Trigger Alpha Factory Refresh", type="primary", key="alpha_refresh_btn"
        ):
            execute_pipeline_live(mode="alpha_refresh")

elif app_mode == "Company Data":
    st.title("Company Data Center")
    st.markdown(
        "Run the mathematically validated pipeline on target companies and monitor forensic outputs."
    )

    st.subheader("Execute ETL Batch")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            "Target Tickers (comma-separated)",
            placeholder="e.g., RELIANCE.NS, TATAPOWER.NS",
            key="etl_ticker_input_widget",
        )
    with col2:
        selected_mode = st.radio(
            "AI Routing Engine:",
            ["Local (Ollama)", "Cloud (Gemini)"],
            horizontal=True,
            key="etl_ai_mode_radio",
        )

    st.caption("Data Spigot Configuration")
    selected_spigot = st.radio(
        "Select Primary Data Source (Auto-Rotate highly recommended to bypass API limits):",
        [
            "Auto-Rotate (FMP -> Alpha Vantage)",
            "Financial Modeling Prep (FMP)",
            "Alpha Vantage (Strictly US)",
            "IndianAPI (Strictly India)",
            "Yahoo Finance",
            "Screener.in",
        ],
        horizontal=True,
        key="etl_spigot_radio",
    )

    spigot_map = {
        "Auto-Rotate (FMP -> Alpha Vantage)": "auto",
        "Financial Modeling Prep (FMP)": "fmp",
        "Alpha Vantage (Strictly US)": "vantage",
        "IndianAPI (Strictly India)": "indianapi",
        "Yahoo Finance": "yfinance",
        "Screener.in": "screener",
    }

    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None

    if st.button(
        "Start ETL Pipeline",
        type="primary",
        use_container_width=False,
        key="etl_start_button",
    ):
        if not ticker_input.strip():
            st.warning("Please enter at least one ticker to proceed.")
        else:
            target_tickers = [
                t.strip().upper() for t in ticker_input.split(",") if t.strip()
            ]
            mode_param = "local" if "Local" in selected_mode else "cloud"
            backend_source = spigot_map[selected_spigot]

            with st.spinner(
                f"Executing pipeline via {selected_spigot} for {len(target_tickers)} client(s)..."
            ):
                try:
                    register_discovered_tickers(target_tickers)
                    st.session_state.pipeline_results = run_etl_pipeline(
                        target_tickers=target_tickers,
                        ai_mode=mode_param,
                        requested_source=backend_source,
                    )
                    st.success("ETL Batch Processing Complete!")
                except Exception as e:
                    st.error(f"Pipeline crashed during execution: {e}")

    if st.session_state.pipeline_results:
        batch_results = st.session_state.pipeline_results
        summary_df = pd.DataFrame(
            [
                {k: v for k, v in res.items() if k != "DataPayload"}
                for res in batch_results
            ]
        )

        st.dataframe(
            summary_df,
            column_config={
                "Ticker": st.column_config.TextColumn("Client (Ticker)"),
                "Status": st.column_config.TextColumn("Fetch Status"),
                "Direct Validation": st.column_config.TextColumn(
                    "Direct Audit (BS/IS/CF)"
                ),
                "Indirect Validation": st.column_config.TextColumn(
                    "Indirect Audit (OCF/FCF)"
                ),
                "Rows Upserted": st.column_config.NumberColumn("Periods Upserted"),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.divider()
        st.subheader("Deep Dive Data Inspector")
        valid_results = [r for r in batch_results if "DataPayload" in r]

        if valid_results:
            inspector_col1, inspector_col2 = st.columns([1, 2])
            with inspector_col1:
                selected_ticker = st.selectbox(
                    "Select Client:",
                    [r["Ticker"] for r in valid_results],
                    key="etl_client_inspector_selectbox",
                )
            with inspector_col2:
                selected_statement = st.radio(
                    "Select Statement:",
                    [
                        "Income Statement (IS)",
                        "Balance Sheet (BS)",
                        "Cash Flow (CF)",
                        "Indirect Cash Flow (ICF)",
                    ],
                    horizontal=True,
                    key="etl_statement_radio",
                )

            stmt_key = (
                "IS"
                if "Income" in selected_statement
                else (
                    "BS"
                    if "Balance" in selected_statement
                    else "ICF" if "Indirect" in selected_statement else "CF"
                )
            )

            client_data = next(
                (
                    item["DataPayload"]
                    for item in valid_results
                    if item["Ticker"] == selected_ticker
                ),
                None,
            )

            if client_data:
                raw_df = client_data[stmt_key]["Raw"]
                clean_df = client_data[stmt_key]["Clean"]

                view_col1, view_col2 = st.columns(2)
                with view_col1:
                    st.caption(f"Raw Extracted Data ({selected_ticker} - {stmt_key})")
                    st.dataframe(raw_df, use_container_width=True)
                with view_col2:
                    st.caption(
                        f"Final DB-Formatted Data ({selected_ticker} - {stmt_key})"
                    )
                    st.dataframe(clean_df, use_container_width=True)


elif app_mode == "Engines":
    st.title("Quantitative Engines & Deep Dive")

    # The repository uses a native DuckDB proxy, not SQLAlchemy.
    try:
        tickers_df = engine.execute(
            'SELECT "Ticker" FROM market_metadata WHERE "IsActive" = TRUE ORDER BY "Ticker"'
        ).df()
        available_db_tickers = (
            tickers_df["Ticker"].dropna().astype(str).tolist()
            if not tickers_df.empty
            else []
        )
    except Exception as e:
        available_db_tickers = []
        st.error(f"Failed to query database: {e}")

    if not available_db_tickers:
        st.info(
            "No active tickers available in the database. Please run the ETL pipeline first."
        )
    else:
        selected_db_ticker = st.selectbox(
            "Select Company to Analyze:",
            available_db_tickers,
            key="single_db_ticker_selectbox",
        )

        edgar_break_date = None
        company_sector = "DEFAULT"
        company_industry = "DEFAULT"

        try:
            profile_df = engine.execute(
                """
                SELECT "valid_data_since", "Sector", "Industry"
                FROM market_metadata
                WHERE "Ticker" = $ticker
                LIMIT 1
                """,
                {"ticker": selected_db_ticker},
            ).df()

            if not profile_df.empty:
                profile_row = profile_df.iloc[0]
                edgar_break_date = profile_row.get("valid_data_since")
                company_sector = profile_row.get("Sector") or "DEFAULT"
                company_industry = profile_row.get("Industry") or "DEFAULT"
        except Exception:
            pass

        st.markdown("### Swarm Memory Controls")
        view_mode = st.radio(
            "Select Historical Perspective:",
            [
                "Swarm Predictive (Strictly Post-Edgar)",
                "Historical Audit (Merged Before & After)",
            ],
            horizontal=True,
        )

        render_company_metrics(
            selected_db_ticker,
            selected_source,
            view_mode,
            edgar_break_date,
            engine,
        )

        st.divider()
        st.markdown("### Engine 1: Market Microstructure")

        latest_market_date = engine.scalar(
            f"""SELECT MAX(date) FROM mv_unified_market_matrix
                WHERE ticker = '{selected_db_ticker.replace("'", "''")}' """
        )

        if latest_market_date is None:
            st.warning(f"No market-matrix data is available for {selected_db_ticker}.")
        else:
            latest_market_date = pd.to_datetime(latest_market_date).date()

            col_date, col_run = st.columns([1, 1])
            with col_date:
                engine_target_date = st.date_input(
                    "Engine 1 Analysis Date",
                    value=latest_market_date,
                    max_value=latest_market_date,
                    key="engine1_target_date",
                )
            with col_run:
                st.write("")
                st.write("")
                run_engine1 = st.button(
                    "Run / Refresh Engine 1",
                    type="primary",
                    use_container_width=True,
                    key="run_engine1_btn",
                )

            if run_engine1:
                with st.spinner(
                    f"Running Engine 1 for {selected_db_ticker} as of {engine_target_date}..."
                ):
                    try:
                        result = OLSMicrostructureEngine().execute_pipeline(
                            ticker=selected_db_ticker,
                            asof_date=str(engine_target_date),
                            lookback=750,
                            run_diagnostics=True,
                        )
                        st.session_state["engine1_last_result"] = result

                        if result.get("status") == "SUCCESS":
                            st.success(
                                f"Engine 1 refreshed successfully for {result['asof_date']}."
                            )
                        else:
                            st.warning(
                                f"Engine 1 completed with status: {result.get('status')}"
                            )
                    except Exception as e:
                        st.error(f"Engine 1 failed: {e}")

            try:
                prediction_df = engine.execute(
                    """
                    SELECT
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
                    WHERE engine_name = '1_Market_Microstructure'
                      AND ticker = $ticker
                      AND asof_date = CAST($asof_date AS DATE)
                    ORDER BY CASE
                        WHEN horizon = '2D' THEN 1
                        WHEN horizon = '5D' THEN 2
                        ELSE 3
                    END
                    """,
                    {
                        "ticker": selected_db_ticker,
                        "asof_date": str(engine_target_date),
                    },
                ).df()
            except Exception as e:
                prediction_df = pd.DataFrame()
                st.error(f"Failed to read Engine 1 predictions: {e}")

            if prediction_df.empty:
                st.info(
                    "No Engine 1 prediction snapshot exists for this date. "
                    "Run / Refresh Engine 1 above."
                )
            else:
                st.subheader("Engine 1 Prediction Snapshot")

                metric_cols = st.columns(len(prediction_df))
                for idx, (_, pred) in enumerate(prediction_df.iterrows()):
                    with metric_cols[idx]:
                        st.metric(
                            f"{pred['horizon']} Direction",
                            str(pred["signal"]),
                        )
                        if pd.notna(pred["score"]):
                            st.caption(f"Direction score: {float(pred['score']):.3f}")
                        if pd.notna(pred["confidence"]):
                            st.caption(
                                f"Walk-forward accuracy: {float(pred['confidence']) * 100:.1f}%"
                            )
                        if pd.notna(pred["data_quality_score"]):
                            st.caption(
                                f"Data quality: {float(pred['data_quality_score']) * 100:.1f}%"
                            )

                selected_horizon = st.selectbox(
                    "Prediction Detail",
                    prediction_df["horizon"].tolist(),
                    key="engine1_horizon_detail",
                )
                selected_pred = prediction_df[
                    prediction_df["horizon"] == selected_horizon
                ].iloc[0]

                try:
                    target_payload = selected_pred["target_metric"]
                    if isinstance(target_payload, str):
                        target_payload = json.loads(target_payload)
                except Exception:
                    target_payload = {}

                try:
                    reason_payload = selected_pred["reason_json"]
                    if isinstance(reason_payload, str):
                        reason_payload = json.loads(reason_payload)
                except Exception:
                    reason_payload = {}

                detail_left, detail_right = st.columns(2)

                with detail_left:
                    st.markdown("#### Expected Return")
                    expected_return = target_payload.get("expected_return")
                    lower = target_payload.get("prediction_lower")
                    upper = target_payload.get("prediction_upper")

                    if expected_return is not None:
                        st.metric(
                            "Model Expected Return",
                            f"{float(expected_return) * 100:.2f}%",
                        )
                    if lower is not None and upper is not None:
                        st.caption(
                            f"Approx. 95% model interval: "
                            f"{float(lower) * 100:.2f}% to {float(upper) * 100:.2f}%"
                        )

                with detail_right:
                    st.markdown("#### Model Diagnostics")
                    model_fit = reason_payload.get("model_fit", {})
                    walk_forward = reason_payload.get("walk_forward", {})
                    st.write(
                        {
                            "r_squared": model_fit.get("r_squared"),
                            "adjusted_r_squared": model_fit.get("adjusted_r_squared"),
                            "training_rows": model_fit.get("training_rows"),
                            "walk_forward_samples": walk_forward.get("sample_size"),
                            "directional_accuracy": walk_forward.get(
                                "directional_accuracy"
                            ),
                            "mae": walk_forward.get("mae"),
                            "rmse": walk_forward.get("rmse"),
                        }
                    )

                st.markdown("#### Current Feature Snapshot")
                try:
                    feature_payload = selected_pred["feature_json"]
                    if isinstance(feature_payload, str):
                        feature_payload = json.loads(feature_payload)
                except Exception:
                    feature_payload = {}

                if feature_payload:
                    feature_df = pd.DataFrame(
                        [
                            {"feature": key, "value": value}
                            for key, value in feature_payload.items()
                        ]
                    )
                    st.dataframe(
                        feature_df,
                        hide_index=True,
                        use_container_width=True,
                    )

        if engine_view == "Canvas Mode (All Engines)":
            st.divider()
            st.markdown("### Canvas Mode")
            st.info(
                "Engine 1 is currently active. Additional engines will be attached "
                "to this canvas after Engine 1 validation."
            )

elif app_mode == "Market Overview":
    st.title("Market Overview")
    st.markdown("Cross-sectional ranking and quadrant analysis.")

    try:
        sectors_df = engine.execute(
            """
            SELECT DISTINCT "Sector"
            FROM market_metadata
            WHERE "Sector" IS NOT NULL
            AND "Sector" <> ''
            ORDER BY "Sector"
            """
        ).df()

        available_sectors = ["All Market"]
        if not sectors_df.empty:
            available_sectors.extend(sectors_df["Sector"].dropna().astype(str).tolist())
    except Exception as e:
        available_sectors = ["All Market"]
        st.error(f"Failed to load sectors: {e}")

    selected_sector = st.selectbox("Filter by Sector", available_sectors)

    if st.button("Generate Sector Analysis", type="primary"):
        with st.spinner("Aggregating live cross-sectional data..."):
            try:
                if selected_sector == "All Market":
                    target_companies = engine.execute(
                        """
                        SELECT "Ticker", "CompanyName"
                        FROM company_profiles
                        WHERE "Ticker" IS NOT NULL
                        ORDER BY "Ticker"
                        """
                    ).df()
                else:
                    target_companies = engine.execute(
                        """
                        SELECT "Ticker", "CompanyName"
                        FROM company_profiles
                        WHERE "Sector" = $sector
                          AND "Ticker" IS NOT NULL
                        ORDER BY "Ticker"
                        """,
                        {"sector": selected_sector},
                    ).df()
            except Exception as e:
                st.error(f"Failed to query sectors: {e}")
                target_companies = pd.DataFrame()

            if target_companies.empty:
                st.warning(
                    "No companies found in this sector. Run the ETL pipeline on more tickers."
                )
            else:
                market_data = []

                for _, row in target_companies.iterrows():
                    t = row["Ticker"]
                    c_name = row["CompanyName"]

                    try:
                        df_roic_tmp = fetch_roic(t, selected_source)
                        df_fcf_tmp = fetch_fcf_yield(t, selected_source)
                    except Exception:
                        continue

                    if not df_roic_tmp.empty and not df_fcf_tmp.empty:
                        latest_roic = (
                            df_roic_tmp["roic"].dropna().iloc[0]
                            if "roic" in df_roic_tmp.columns
                            and not df_roic_tmp["roic"].dropna().empty
                            else None
                        )
                        latest_fcf = (
                            df_fcf_tmp["FCF_Yield"].dropna().iloc[0]
                            if "FCF_Yield" in df_fcf_tmp.columns
                            and not df_fcf_tmp["FCF_Yield"].dropna().empty
                            else None
                        )

                        if latest_roic is not None and latest_fcf is not None:
                            market_data.append(
                                {
                                    "Ticker": t,
                                    "Company": c_name,
                                    "ROIC (%)": latest_roic * 100,
                                    "FCF Yield (%)": latest_fcf * 100,
                                }
                            )

                market_df = pd.DataFrame(market_data)

                if not market_df.empty:
                    st.divider()
                    st.subheader("FCF Yield vs ROIC")

                    st.scatter_chart(
                        market_df,
                        x="FCF Yield (%)",
                        y="ROIC (%)",
                        color="Ticker",
                        height=500,
                    )

                    st.subheader("Sector Leaderboard")
                    leaderboard = market_df.sort_values(
                        by="ROIC (%)", ascending=False
                    ).reset_index(drop=True)
                    st.dataframe(leaderboard, use_container_width=True)
                else:
                    st.warning(
                        "Not enough clean ratio data to generate the quadrant map."
                    )
