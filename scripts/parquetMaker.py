from pathlib import Path

import duckdb

from database import DB_PATH

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ==========================================================
# CONFIGURATION
# ==========================================================

PARQUET_ROOT = Path("parquet")

ARCHIVE_ROOT = PARQUET_ROOT / "archive"
INCREMENTAL_ROOT = PARQUET_ROOT / "incremental"

ARCHIVE_DRIVE_FOLDER_ID = "12uX5qeQOVchjKKImC4NpSGn8RZ-LLpL-"

# Tables to migrate
TABLES = {
    "unified_market_master": "ReportDate",
    "global_assets_daily": "ReportDate",
    "global_assets_intraday": "ReportDate",
    "macro_daily_ledger": "ReportDate",
    "macro_intraday_ledger": "ReportDate",
    "institutional_ledger": "ReportDate",
    "trade_events_ledger": "ReportDate",
}


class ParquetMaker:

    def __init__(self):

        self.con = duckdb.connect(DB_PATH, read_only=True)

        ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        INCREMENTAL_ROOT.mkdir(parents=True, exist_ok=True)
        self.connect_drive()

    def _run(self, query: str):
        """Execute SQL."""

        self.con.execute(query)

    def _scalar(self, query: str):
        """Return a single scalar."""

        result = self.con.execute(query).fetchone()

        if result is None:
            return None

        return result[0]

    def _table_exists(self, table: str):

        query = f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name='{table}'
        """

        count = self._scalar(query)

        return count is not None and count > 0

    def _get_years(self, table, date_column):

        query = f"""
        SELECT DISTINCT YEAR("{date_column}")
        FROM {table}
        ORDER BY 1
        """

        return [row[0] for row in self.con.execute(query).fetchall()]

    def _latest_partition(self, table, date_column):

        query = f"""
        SELECT
            YEAR(MAX("{date_column}")),
            MONTH(MAX("{date_column}"))
        FROM {table}
        """

        return self.con.execute(query).fetchone()

    def _archive_path(self, table, year):

        path = ARCHIVE_ROOT / table / f"year{year}"

        path.mkdir(parents=True, exist_ok=True)

        return path

    def _incremental_path(self, table, year):

        path = INCREMENTAL_ROOT / table / f"yearIncremental{year}"

        path.mkdir(parents=True, exist_ok=True)

        return path

    def _export_year(self, table, date_column, year):

        output_dir = self._archive_path(table, year)
        output_file = output_dir / "data.parquet"

        print(f"[+] Exporting {table} ({year})")

        if output_file.exists():
            output_file.unlink()

        query = f"""
        COPY (
            SELECT *
            FROM {table}
            WHERE YEAR("{date_column}") = {year}
        )
        TO '{output_file.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        );
        """

        self._run(query)

        print(f"    ✓ {output_file}")

    def create_archive_parquet(self):

        print("=" * 70)
        print("Creating Historical Archive")
        print("=" * 70)

        for table, date_column in TABLES.items():

            print(f"\nProcessing : {table}")

            if not self._table_exists(table):
                print("    Table not found.")
                continue

            years = self._get_years(table, date_column)

            if not years:
                print("    No data.")
                continue

            for year in years:
                self._export_year(table, date_column, year)

        print("\nArchive generation completed.")

    def _export_month(self, table, date_column, year, month):

        output_dir = self._incremental_path(table, year)

        output_file = output_dir / f"{month:02d}.parquet"

        print(f"[+] Exporting {table} ({year}-{month:02d})")

        if output_file.exists():
            output_file.unlink()

        query = f"""
        COPY (
            SELECT *
            FROM {table}
            WHERE YEAR("{date_column}") = {year}
            AND MONTH("{date_column}") = {month}
        )
        TO '{output_file.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        );
        """

        self._run(query)

        print(f"    ✓ {output_file}")

    def create_incremental_parquet(self):

        print("=" * 70)
        print("Creating Incremental Parquet")
        print("=" * 70)

        for table, date_column in TABLES.items():

            print(f"\nProcessing : {table}")

            if not self._table_exists(table):
                print("    Table not found.")
                continue

            latest = self._latest_partition(table, date_column)

            if latest is None:
                print("    No data.")
                continue

            year, month = latest

            if year is None or month is None:
                print("    No valid dates.")
                continue

            self._export_month(table, date_column, year, month)

        print("\nIncremental export completed.")

    def connect_drive(self):

        creds = Credentials.from_authorized_user_file(
            "token.json",
            ["https://www.googleapis.com/auth/drive"],
        )

        self.drive = build(
            "drive",
            "v3",
            credentials=creds,
        )

    def _get_or_create_folder(self, name: str, parent_id: str):

        query = (
            f"name='{name}' and "
            f"'{parent_id}' in parents and "
            "mimeType='application/vnd.google-apps.folder' and "
            "trashed=false"
        )

        result = self.drive.files().list(q=query, fields="files(id,name)").execute()

        folders = result.get("files", [])

        if folders:
            return folders[0]["id"]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }

        folder = (
            self.drive.files()
            .create(
                body=metadata,
                fields="id",
            )
            .execute()
        )

        return folder["id"]

    def _upload_file(self, local_file: Path, folder_id: str):

        media = MediaFileUpload(
            str(local_file),
            resumable=True,
        )

        metadata = {
            "name": local_file.name,
            "parents": [folder_id],
        }

        self.drive.files().create(
            body=metadata,
            media_body=media,
            fields="id",
        ).execute()

        print(f"[+] Uploaded {local_file}")

    def _upload_directory(self, local_root: Path, drive_root_id: str):

        for file in local_root.rglob("*.parquet"):

            relative = file.relative_to(local_root)

            parent = drive_root_id

            for folder in relative.parts[:-1]:

                parent = self._get_or_create_folder(
                    folder,
                    parent,
                )

            self._upload_file(
                file,
                parent,
            )

    def upload_archive_to_drive(self, folder_id):

        print("=" * 70)
        print("Uploading Archive")
        print("=" * 70)

        self._upload_directory(
            ARCHIVE_ROOT,
            folder_id,
        )

        print("\nArchive upload completed.")

    def upload_incremental_to_drive(self, folder_id):

        print("=" * 70)
        print("Uploading Incremental")
        print("=" * 70)

        self._upload_directory(
            INCREMENTAL_ROOT,
            folder_id,
        )

        print("\nIncremental upload completed.")

    def close(self):
        self.con.close()


if __name__ == "__main__":

    pm = ParquetMaker()

    try:
        # Uncomment ONE operation at a time

        # pm.create_archive_parquet()

        # pm.create_incremental_parquet()

        # pm.upload_archive_to_drive(ARCHIVE_DRIVE_FOLDER_ID)

        pm.upload_incremental_to_drive(ARCHIVE_DRIVE_FOLDER_ID)

        pass

    finally:
        pm.close()
