<h1 align="center">Financial ETL & Analysis Architecture</h1>

<h2>System Architecture Diagram</h2>
<pre><code>
=============================================================================
                      company_financial_data_ETL
=============================================================================

+---------------------------------------------------------------------------+
|                             DATA SOURCES (EXTRACT)                        |
|  [ NSE Archives ]  [ EDGAR ]  [ Yahoo Finance ]  [ Screener ] [ Vantage ] |
+------------------------------------+--------------------------------------+
                                     |
+------------------------------------v--------------------------------------+
|                         ORCHESTRATION & SCRAPING                          |
|  +---------------------------------------------------------------------+  |
|  | - downloadOrchestrator.py      - statementScrape.py                 |  |
|  | - nseScrape.py                 - macroScrape.py                     |  |
|  | - nseArchiveLooper.py          - edgarUtils.py                      |  |
|  +---------------------------------------------------------------------+  |
+------------------------------------+--------------------------------------+
                                     | Raw Data / JSONs (offline_statements)
+------------------------------------v--------------------------------------+
|                        DATA TRANSFORMATION (TRANSFORM)                    |
|  +-----------------------+  +--------------------+  +------------------+  |
|  |     Ingestion         |  |   Processing       |  |  Validation      |  |
|  | - ingestUnifiedMatrix |  | - ratioAnalysis.py |  | - reconciliation |  |
|  | - ingestEvents.py     |  | - fiiDiiBackfill   |  |                  |  |
|  | - mapping_config.json |  |                    |  |                  |  |
|  +-----------------------+  +--------------------+  +------------------+  |
+------------------------------------+--------------------------------------+
                                     | 
+------------------------------------v--------------------------------------+
|                            DATA STORAGE (LOAD)                            |
|  +---------------------------------------------------------------------+  |
|  | - database.py                                                       |  |
|  | - materializedViewEngine.py                                         |  |
|  | - PostgreSQL / DuckDB (migrate_pg_to_duckdb.py)                     |  |
|  +---------------------------------------------------------------------+  |
+------------------------------------+--------------------------------------+
                                     |
+------------------------------------v--------------------------------------+
|                      ANALYTICS, AI & UI (CONSUMPTION)                     |
|  +-----------------------+  +--------------------+  +------------------+  |
|  |      AI & LLM         |  |   Analytic Engines |  |    Interfaces    |  |
|  | - ai_agent.py         |  | - olsEngine1.py    |  | - dashboard.py   |  |
|  | - reasoning.py        |  | - auditorEngine.py |  | - olsEngine1UI   |  |
|  | - vectorize.py        |  | - companyMetrics   |  | - Jupyter NBs    |  |
|  +-----------------------+  +--------------------+  +------------------+  |
+---------------------------------------------------------------------------+
</code></pre>

<h2>Project Overview</h2>
<p>The <strong>Financial ETL Project</strong> is a comprehensive backend system designed to extract, transform, and load financial market data, corporate statements, and macroeconomic indicators. It heavily features an automated Python pipeline that integrates traditional quantitative analytics (like OLS regression and ratio analysis) with modern Large Language Model (LLM) components for data validation, reasoning, and vectorization.</p>

<h2>1. Extraction Layer (Scraping & Orchestration)</h2>
<p>The extraction layer handles the acquisition of raw data from multiple financial sources, utilizing scheduled scrapers and API fetchers.</p>
<ul>
  <li><strong>Orchestration:</strong> Controlled by <code>downloadOrchestrator.py</code>, ensuring that sequential data downloads do not hit rate limits.</li>
  <li><strong>Market & Macro Scrapers:</strong> Dedicated scripts fetch data from the National Stock Exchange (<code>nseScrape.py</code>, <code>nseArchiveLooper.py</code>), macroeconomic sources (<code>macroScrape.py</code>), and regulatory filings (<code>edgarUtils.py</code>).</li>
  <li><strong>Statement Fetching:</strong> <code>statementScrape.py</code> pulls balance sheets, cash flows, and income statements across multiple providers (Screener, Yahoo Finance, Vantage). The raw outputs are stored in the <code>offline_statements/</code> directory as JSON files for staging.</li>
</ul>

<h2>2. Transformation Layer (Processing)</h2>
<p>This layer cleans, normalizes, and calculates derived metrics from the raw financial JSON data.</p>
<ul>
  <li><strong>Standardization:</strong> <code>mapping_config.json</code> acts as the translation layer to align disparate terminology across APIs (e.g., matching "Total Revenue" from Yahoo Finance to "Sales" from Screener).</li>
  <li><strong>Data Ingestion:</strong> Scripts like <code>ingestUnifiedMatrix.py</code>, <code>ingestEvents.py</code>, and <code>ingestInstitutional.py</code> process normalized data arrays into relational formats.</li>
  <li><strong>Financial Logic:</strong> <code>ratioAnalysis.py</code> calculates fundamental metrics, while <code>reconciliation.py</code> audits and balances the math to ensure data integrity prior to database commits.</li>
</ul>

<h2>3. Load & Storage Layer (Database)</h2>
<p>The data layer supports heavy read/write operations utilizing both row-oriented and column-oriented paradigms.</p>
<ul>
  <li><strong>Core Connections:</strong> Handled by <code>database.py</code> for primary SQL interactions.</li>
  <li><strong>Optimization:</strong> The <code>materializedViewEngine.py</code> pre-computes complex joins to speed up downstream analytics.</li>
  <li><strong>Migration & Scaling:</strong> The presence of <code>migrate_pg_to_duckdb.py</code> indicates a hybrid storage approach, utilizing PostgreSQL for robust transactional data and DuckDB for fast, in-memory analytical processing.</li>
</ul>

<h2>4. Analytics, AI & Consumption Layer</h2>
<p>The top layer applies quantitative models and AI reasoning to the structured data, visualizing it for the end user.</p>
<ul>
  <li><strong>Analytical Engines:</strong> <code>engines/</code> houses statistical models. <code>olsEngine1.py</code> handles Ordinary Least Squares regressions for forecasting, <code>companyMetrics.py</code> tracks fundamental health, and <code>auditorEngine.py</code> verifies anomalies.</li>
  <li><strong>AI Integration:</strong> Scripts like <code>ai_agent.py</code>, <code>reasoning.py</code>, and <code>vectorize.py</code> embed AI directly into the pipeline, allowing natural language queries against financial data and intelligent anomaly detection.</li>
  <li><strong>Interfaces:</strong> Users interact with the data via Python-based interfaces such as <code>dashboard.py</code>, specialized UI scripts (<code>olsEngine1UI.py</code>), and exploratory Jupyter Notebooks (<code>pandas.ipynb</code>, <code>statementScrape.ipynb</code>).
    
  RUN -

  streamlit run dashboard.py
  </li>
</ul>


