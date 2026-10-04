import os
import re
import time
from datetime import date, datetime
from html import unescape

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = "offline_data_cache/master_archives"
SOURCE_URL = "https://www.arihantcapital.com/derivatives/fii-dii-trading-activities"
OUTPUT_FILE = "niftytrader_fiidii_master.csv"

DEFAULT_START_DATE = date(2015, 1, 1)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================
# BASIC HELPERS
# ============================================================================


def clean_number(value):
    """
    Convert Arihant table values such as:
        12,345.67
        ₹12,345.67
        -123.45
    into float.
    """

    if value is None:
        return None

    value = str(value).strip().replace("\xa0", " ")

    if not value:
        return None

    value = value.replace("₹", "").replace(",", "").strip()

    try:
        return float(value)
    except ValueError:
        return None


def parse_hidden_inputs(html):
    """
    Extract ASP.NET hidden fields from a normal HTML response.

    This is important because Arihant uses WebForms and requires
    __VIEWSTATE and related fields to be carried across postbacks.
    """

    soup = BeautifulSoup(unescape(html), "html.parser")

    state = {}

    for tag in soup.select('input[type="hidden"]'):
        name = tag.get("name")

        if name:
            state[name] = tag.get("value", "")

    return state


# ============================================================================
# ASP.NET AJAX DELTA RESPONSE PARSER
# ============================================================================


def parse_delta_response(text, old_state):
    """
    Parse Arihant's ASP.NET AJAX delta response.

    Example structure:

        17442|updatePanel|panel-id|<html...>
        2752|hiddenField|__VIEWSTATE|<value...>

    IMPORTANT:
    The length values refer to the raw response content.

    Therefore we do NOT unescape the complete response before slicing,
    because converting things like &#39; -> ' changes the character count
    and can corrupt the length-based parsing.
    """

    raw_text = text

    panel_html = None
    updated_state = dict(old_state)

    # ----------------------------------------------------------------------
    # Parse updatePanel
    # ----------------------------------------------------------------------

    panel_pattern = re.compile(
        r"(?:^|\|)(\d+)\|updatePanel\|([^|]+)\|",
        flags=re.IGNORECASE,
    )

    panel_match = panel_pattern.search(raw_text)

    if panel_match:
        length = int(panel_match.group(1))
        start = panel_match.end()

        panel_html = raw_text[start : start + length]

    # ----------------------------------------------------------------------
    # Parse hidden fields
    # ----------------------------------------------------------------------

    hidden_pattern = re.compile(
        r"(?:^|\|)(\d+)\|hiddenField\|([^|]+)\|",
        flags=re.IGNORECASE,
    )

    position = 0

    while True:
        match = hidden_pattern.search(raw_text, position)

        if not match:
            break

        length = int(match.group(1))
        field_name = match.group(2)

        start = match.end()

        field_value = raw_text[start : start + length]

        updated_state[field_name] = unescape(field_value)

        position = start + length

    # ----------------------------------------------------------------------
    # Some responses may be ordinary HTML instead of an AJAX delta
    # ----------------------------------------------------------------------

    if panel_html is None:
        panel_html = raw_text

        normal_state = parse_hidden_inputs(panel_html)

        if normal_state:
            updated_state.update(normal_state)

    return unescape(panel_html), updated_state


# ============================================================================
# TABLE PARSER
# ============================================================================


def parse_activity_table(html):
    """
    Extract Arihant FII/DII activity table.

    Expected columns:

        Date
        Gross Purchase
        Gross Sales
        Net Purchase/Sales

    Returns:

        [
            {
                "date": "...",
                "buy": "...",
                "sell": "...",
                "net": "..."
            }
        ]
    """

    soup = BeautifulSoup(unescape(html), "html.parser")

    for table in soup.find_all("table"):

        headers = [
            cell.get_text(" ", strip=True).lower() for cell in table.find_all("th")
        ]

        if not headers:
            continue

        has_required_columns = (
            any(h == "date" for h in headers)
            and any("gross purchase" in h for h in headers)
            and any("gross sales" in h for h in headers)
            and any("net purchase/sales" in h for h in headers)
        )

        if not has_required_columns:
            continue

        date_i = next(i for i, h in enumerate(headers) if h == "date")

        buy_i = next(i for i, h in enumerate(headers) if "gross purchase" in h)

        sell_i = next(i for i, h in enumerate(headers) if "gross sales" in h)

        net_i = next(i for i, h in enumerate(headers) if "net purchase/sales" in h)

        rows = []

        for tr in table.find_all("tr"):

            cells = tr.find_all("td")

            if not cells:
                continue

            max_index = max(
                date_i,
                buy_i,
                sell_i,
                net_i,
            )

            if len(cells) <= max_index:
                continue

            rows.append(
                {
                    "date": cells[date_i].get_text(
                        " ",
                        strip=True,
                    ),
                    "buy": cells[buy_i].get_text(
                        " ",
                        strip=True,
                    ),
                    "sell": cells[sell_i].get_text(
                        " ",
                        strip=True,
                    ),
                    "net": cells[net_i].get_text(
                        " ",
                        strip=True,
                    ),
                }
            )

        return rows

    return []


# ============================================================================
# PAGINATION
# ============================================================================


def current_page_number(html):
    """
    Determine the active page number from Arihant's pagination markup.
    """

    soup = BeautifulSoup(
        unescape(html),
        "html.parser",
    )

    container = soup.select_one(".custom-pagination")

    if container is None:
        return 1

    active = container.select_one(".active")

    if active is None:
        return 1

    label = active.get_text(
        " ",
        strip=True,
    )

    if label.isdigit():
        return int(label)

    return 1


def find_next_page(html, current_page):
    """
    Find Arihant's forward pagination control.

    Arihant's pager uses:

        ctl01$ctlXX -> numbered page buttons
        ctl02$ctl00 -> NEXT / forward arrow
        ctl00$ctl00 -> PREVIOUS / back arrow

    Do not try to infer the next page from the numbered buttons.
    The numbered button targets are positional and can point backwards
    after the pagination window shifts.
    """

    soup = BeautifulSoup(
        unescape(html),
        "html.parser",
    )

    container = soup.select_one(".custom-pagination")

    if container is None:
        return None

    anchors = container.find_all("a")

    postback_pattern = re.compile(
        r"__doPostBack\(\s*" r"['\"]([^'\"]+)['\"]\s*,\s*" r"['\"]([^'\"]*)['\"]\s*\)",
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # ALWAYS prefer Arihant's actual forward-arrow control.
    #
    # This is the control seen in the captured HTML:
    #
    # ctl00$ContentPlaceHolder1$dtpgrGain$ctl02$ctl00
    # ------------------------------------------------------------------

    for anchor in anchors:

        match = postback_pattern.search(anchor.get("href", ""))

        if not match:
            continue

        target = match.group(1)
        argument = match.group(2)

        if target.endswith("$dtpgrGain$ctl02$ctl00"):
            return (
                current_page + 1,
                target,
                argument,
            )

    # ------------------------------------------------------------------
    # Fallback to the ellipsis control if the forward arrow is not
    # present.
    # ------------------------------------------------------------------

    for anchor in anchors:

        label = anchor.get_text(
            " ",
            strip=True,
        )

        if label not in ("...", "…"):
            continue

        match = postback_pattern.search(anchor.get("href", ""))

        if not match:
            continue

        target = match.group(1)
        argument = match.group(2)

        if "$dtpgrGain$" in target:
            return (
                current_page + 1,
                target,
                argument,
            )

    return None


# ============================================================================
# HTTP / ASP.NET TRANSPORT
# ============================================================================


def get_initial_page(session):
    """
    Load a fresh Arihant page and collect all ASP.NET hidden state.
    """

    print(
        "    [1/4] Connecting to Arihant...",
        flush=True,
    )

    response = session.get(
        SOURCE_URL,
        timeout=(10, 30),
        allow_redirects=True,
    )

    response.raise_for_status()

    if not response.text:
        raise RuntimeError("Arihant returned an empty page.")

    state = parse_hidden_inputs(response.text)

    if "__VIEWSTATE" not in state:

        raise RuntimeError(
            "Arihant page loaded, but __VIEWSTATE was not found. "
            "The website structure may have changed."
        )

    print(
        f"        connected: HTTP {response.status_code}, "
        f"{len(response.text):,} bytes, "
        f"{len(state)} hidden fields",
        flush=True,
    )

    return response.text, state


def postback(
    session,
    state,
    fields,
    target,
    argument="",
):
    """
    Perform an ASP.NET AJAX postback.

    Carries forward:

        __VIEWSTATE
        __VIEWSTATEGENERATOR
        __EVENTVALIDATION
        other hidden fields

    and adds the requested event target.
    """

    payload = dict(state)

    payload.update(fields)

    payload["__EVENTTARGET"] = target
    payload["__EVENTARGUMENT"] = argument
    payload["__LASTFOCUS"] = ""

    payload["ctl00$scrptmanagr"] = (
        "ctl00$ContentPlaceHolder1$UpdatePanelBigSch|" + target
    )

    payload["__ASYNCPOST"] = "true"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Referer": SOURCE_URL,
        "Origin": "https://www.arihantcapital.com",
        "Content-Type": ("application/x-www-form-urlencoded; " "charset=UTF-8"),
        "X-Requested-With": "XMLHttpRequest",
        "X-MicrosoftAjax": "Delta=true",
    }

    response = session.post(
        SOURCE_URL,
        data=payload,
        headers=headers,
        timeout=(10, 30),
    )

    response.raise_for_status()

    html, new_state = parse_delta_response(
        response.text,
        state,
    )

    if not html:
        raise RuntimeError("Arihant returned an empty postback response.")

    return html, new_state


# ============================================================================
# CATEGORY SCRAPER
# ============================================================================


def fetch_category(
    session,
    start_date,
    end_date,
    category,
):
    """
    Fetch one complete FII/FPI or DII date range.

    Handles ALL pagination pages.
    """

    print(
        f"        [2/4] Requesting {category}: "
        f"{start_date:%Y-%m-%d} -> "
        f"{end_date:%Y-%m-%d}",
        flush=True,
    )

    # ----------------------------------------------------------------------
    # Fresh ASP.NET session state for each category
    # ----------------------------------------------------------------------

    html, state = get_initial_page(session)

    # ----------------------------------------------------------------------
    # Form values taken from Arihant's actual request
    # ----------------------------------------------------------------------

    fields = {
        "ctl00$ContentPlaceHolder1$cattypeid": "cash",
        "ctl00$ContentPlaceHolder1$fosubCatid": "index",
        "ctl00$ContentPlaceHolder1$ddlSubCategory": category,
        "ctl00$ContentPlaceHolder1$fromdate": (start_date.strftime("%Y-%m-%d")),
        "ctl00$ContentPlaceHolder1$todate": (end_date.strftime("%Y-%m-%d")),
        "search2": "",
        # These are present in Arihant's normal page/form state.
        "ctl00$ContentPlaceHolder1$DrpMobmenu": "DM",
        "ctl00$ContentPlaceHolder1$menuDM": ("/derivatives/fii-dii-trading-activities"),
    }

    # ----------------------------------------------------------------------
    # Submit Search / Go
    # ----------------------------------------------------------------------

    html, state = postback(
        session,
        state,
        fields,
        "ctl00$ContentPlaceHolder1$btnGo",
    )

    records = []

    current_page = current_page_number(html)

    seen_pages = {current_page}

    print(
        f"        starting at pagination page " f"{current_page}",
        flush=True,
    )

    # =========================================================================
    # PAGINATION LOOP
    # =========================================================================

    while True:

        # ------------------------------------------------------------------
        # Parse current page
        # ------------------------------------------------------------------

        page_rows = parse_activity_table(html)

        rows_added = 0

        for row in page_rows:

            parsed_date = pd.to_datetime(
                row.get("date"),
                format="%d-%b-%Y",
                errors="coerce",
            )

            if pd.isna(parsed_date):
                continue

            row_date = parsed_date.date()

            # Respect requested date boundaries.
            if not (start_date <= row_date <= end_date):
                continue

            records.append(
                {
                    "date": row_date.isoformat(),
                    "buy": clean_number(row.get("buy")),
                    "sell": clean_number(row.get("sell")),
                    "net": clean_number(row.get("net")),
                }
            )

            rows_added += 1

        print(
            f"        page {current_page}: " f"{rows_added} rows",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Find forward pagination control
        # ------------------------------------------------------------------

        next_page_info = find_next_page(
            html,
            current_page,
        )

        if next_page_info is None:

            print(
                f"        no further pagination after " f"page {current_page}",
                flush=True,
            )

            break

        expected_page, target, argument = next_page_info

        print(
            f"        requesting next page after "
            f"{current_page} "
            f"(expected {expected_page})",
            flush=True,
        )

        # ------------------------------------------------------------------
        # IMPORTANT:
        # Keep the current ASP.NET state.
        #
        # The VIEWSTATE returned from the previous page must be sent into
        # the next postback.
        # ------------------------------------------------------------------

        html, state = postback(
            session,
            state,
            fields,
            target,
            argument,
        )

        # ------------------------------------------------------------------
        # Trust server's returned active page, NOT our expected page.
        # ------------------------------------------------------------------

        new_page = current_page_number(html)

        if new_page <= current_page:
            print(
                f"        no further pages after page {current_page}",
                flush=True,
            )
            break

        if new_page in seen_pages:

            raise RuntimeError("Pagination loop detected at " f"page {new_page}.")

        seen_pages.add(new_page)

        current_page = new_page

        # Small delay to avoid hammering the site.
        time.sleep(0.15)

    # ----------------------------------------------------------------------
    # Remove any duplicate dates caused by page boundaries
    # ----------------------------------------------------------------------

    by_date = {}

    for row in records:
        by_date[row["date"]] = row

    records = sorted(
        by_date.values(),
        key=lambda row: row["date"],
    )

    print(
        f"        [3/4] {category}: " f"extracted {len(records):,} daily rows",
        flush=True,
    )

    return records


# ============================================================================
# YEAR RANGE GENERATOR
# ============================================================================


def year_ranges(
    start_date,
    end_date,
):
    """
    Split requested date range into calendar-year chunks.

    Example:

        2025-12-20 -> 2026-01-10

    becomes:

        2025-12-20 -> 2025-12-31
        2026-01-01 -> 2026-01-10
    """

    for year in range(
        start_date.year,
        end_date.year + 1,
    ):

        range_start = max(
            start_date,
            date(year, 1, 1),
        )

        range_end = min(
            end_date,
            date(year, 12, 31),
        )

        yield (
            range_start,
            range_end,
        )


# ============================================================================
# DATASET BUILDER
# ============================================================================


def build_dataset(
    start_date,
    end_date,
):
    """
    Build combined FII/DII dataset.
    """

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": ("en-US,en;q=0.9"),
            "Connection": "keep-alive",
        }
    )

    fii_all = []
    dii_all = []

    ranges = list(
        year_ranges(
            start_date,
            end_date,
        )
    )

    print(
        f"    Splitting request into " f"{len(ranges)} calendar-year range(s).",
        flush=True,
    )

    # =========================================================================
    # YEAR LOOP
    # =========================================================================

    for index, (
        range_start,
        range_end,
    ) in enumerate(
        ranges,
        start=1,
    ):

        print(
            f"\n    YEAR {index}/{len(ranges)}: "
            f"{range_start:%Y-%m-%d} -> "
            f"{range_end:%Y-%m-%d}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # FII/FPI
        # ------------------------------------------------------------------

        fii_records = fetch_category(
            session,
            range_start,
            range_end,
            "FII/FPI",
        )

        fii_all.extend(fii_records)

        time.sleep(0.5)

        # ------------------------------------------------------------------
        # DII
        # ------------------------------------------------------------------

        dii_records = fetch_category(
            session,
            range_start,
            range_end,
            "DII",
        )

        dii_all.extend(dii_records)

        time.sleep(0.5)

    # =========================================================================
    # DATAFRAME CREATION
    # =========================================================================

    fii_df = pd.DataFrame(
        fii_all,
        columns=[
            "date",
            "buy",
            "sell",
            "net",
        ],
    )

    dii_df = pd.DataFrame(
        dii_all,
        columns=[
            "date",
            "buy",
            "sell",
            "net",
        ],
    )

    if fii_df.empty and dii_df.empty:
        return pd.DataFrame()

    # =========================================================================
    # RENAME FII
    # =========================================================================

    if not fii_df.empty:

        fii_df = fii_df.rename(
            columns={
                "buy": "fii_buy",
                "sell": "fii_sell",
                "net": "fii_net",
            }
        )

    # =========================================================================
    # RENAME DII
    # =========================================================================

    if not dii_df.empty:

        dii_df = dii_df.rename(
            columns={
                "buy": "dii_buy",
                "sell": "dii_sell",
                "net": "dii_net",
            }
        )

    # =========================================================================
    # MERGE
    # =========================================================================

    if fii_df.empty:

        merged = dii_df

    elif dii_df.empty:

        merged = fii_df

    else:

        merged = pd.merge(
            fii_df,
            dii_df,
            on="date",
            how="outer",
        )

    # ----------------------------------------------------------------------
    # Arihant does not provide Nifty close in this table.
    #
    # Keep the expected downstream column so ingestInstitutional.py sees
    # the same CSV schema as before.
    # ----------------------------------------------------------------------

    merged["nifty_close"] = None

    # =========================================================================
    # FINAL COLUMN ORDER
    # =========================================================================

    columns = [
        "date",
        "fii_buy",
        "fii_sell",
        "fii_net",
        "dii_buy",
        "dii_sell",
        "dii_net",
        "nifty_close",
    ]

    for column in columns:

        if column not in merged.columns:
            merged[column] = None

    merged = (
        merged[columns]
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return merged


# ============================================================================
# MAIN BACKFILL FUNCTION
# ============================================================================


def run_fiidii_backfill(
    years_to_fetch=None,
    start_date=None,
    end_date=None,
):
    """
    Public backfill function.

    Supports both:

        run_fiidii_backfill()

    and:

        run_fiidii_backfill(
            start_date="2026-09-29",
            end_date="2026-10-01"
        )

    The latter is what the orchestrator should use.
    """

    print(
        "=== Starting Arihant FII/DII Backfill ===",
        flush=True,
    )

    try:

        # ==================================================================
        # Convert provided dates
        # ==================================================================

        if start_date is not None:

            start_date = pd.to_datetime(start_date).date()

        if end_date is not None:

            end_date = pd.to_datetime(end_date).date()

        # ==================================================================
        # Backward-compatible years_to_fetch support
        # ==================================================================

        if start_date is None and end_date is None and years_to_fetch is not None:

            years = sorted(set(int(year) for year in years_to_fetch))

            if not years:

                print(
                    "[!] No years supplied.",
                    flush=True,
                )

                return

            start_date = date(
                min(years),
                1,
                1,
            )

            end_date = date(
                max(years),
                12,
                31,
            )

        # ==================================================================
        # Defaults
        # ==================================================================

        if start_date is None:
            start_date = DEFAULT_START_DATE

        if end_date is None:
            end_date = datetime.now().date()

        # ==================================================================
        # Validate
        # ==================================================================

        if start_date > end_date:

            raise ValueError("start_date cannot be after end_date")

        print(
            f"    Date range: " f"{start_date:%Y-%m-%d} -> " f"{end_date:%Y-%m-%d}",
            flush=True,
        )

        # ==================================================================
        # SCRAPE
        # ==================================================================

        df = build_dataset(
            start_date,
            end_date,
        )

        if df.empty:

            print(
                "\n[!] No FII/DII data was extracted.",
                flush=True,
            )

            return

        # ==================================================================
        # SAVE
        # ==================================================================

        file_path = os.path.join(
            CACHE_DIR,
            OUTPUT_FILE,
        )

        df.to_csv(
            file_path,
            index=False,
        )

        print(
            "\n[4/4] COMPLETE",
            flush=True,
        )

        print(
            f"    Saved: {file_path}",
            flush=True,
        )

        print(
            f"    Rows: {len(df):,}",
            flush=True,
        )

        print(
            f"    Range: " f"{df['date'].min()} -> " f"{df['date'].max()}",
            flush=True,
        )

    except requests.RequestException as exc:

        print(
            "\n[X] HTTP/network error while " f"contacting Arihant: {exc}",
            flush=True,
        )

        raise

    except Exception as exc:

        print(
            f"\n[X] Arihant scraper failed: {exc}",
            flush=True,
        )

        raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=("Backfill daily FII/DII cash activity " "from Arihant Capital.")
    )

    parser.add_argument(
        "--start",
        type=str,
        help="Start date YYYY-MM-DD",
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date YYYY-MM-DD",
    )

    args = parser.parse_args()

    # ========================================================================
    # ORCHESTRATOR / DELTA MODE
    # ========================================================================

    if args.start and args.end:

        print(
            "=== Running Delta Sync for FII/DII: " f"{args.start} to {args.end} ===",
            flush=True,
        )

        run_fiidii_backfill(
            start_date=args.start,
            end_date=args.end,
        )

    # ========================================================================
    # Invalid partial arguments
    # ========================================================================

    elif args.start or args.end:

        parser.error("--start and --end must be supplied together")

    # ========================================================================
    # FULL BACKFILL MODE
    # ========================================================================

    else:

        run_fiidii_backfill()
