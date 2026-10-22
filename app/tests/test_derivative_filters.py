from __future__ import annotations

import sqlite3
import unittest

from app.services.derivative_filters import (
    COMMODITY_FUTURES_SQL,
    COMMODITY_OPTIONS_SQL,
    STOCK_INDEX_FUTURES_SQL,
    STOCK_INDEX_OPTIONS_SQL,
    classify_contract,
)


class DerivativeInstrumentFilterTests(unittest.TestCase):
    def test_futcom_is_explicitly_selected_as_commodity_future(self):
        self.assertEqual(classify_contract("FUTCOM", "XX"), "commodity_future")
        self.assertEqual(classify_contract("futcom", None), "commodity_future")
        self.assertIn("'FUTCOM'", COMMODITY_FUTURES_SQL)

    def test_commodity_options_are_not_misclassified_as_futures(self):
        self.assertEqual(classify_contract("OPTFUT", "CE"), "commodity_option")
        self.assertEqual(classify_contract("OPTFUT", "PE"), "commodity_option")
        self.assertEqual(classify_contract("OPTCOM", "CE"), "commodity_option")
        self.assertEqual(classify_contract("OPTFUT", "XX"), "other")
        self.assertNotIn("'OPTFUT'", COMMODITY_FUTURES_SQL)
        self.assertIn("'OPTFUT'", COMMODITY_OPTIONS_SQL)

    def test_stock_index_contracts_remain_in_separate_sources(self):
        self.assertEqual(classify_contract("FUTSTK", "XX"), "stock_index_future")
        self.assertEqual(classify_contract("OPTSTK", "CE"), "stock_index_option")
        self.assertEqual(classify_contract("FUTSTK", "CE"), "other")
        self.assertEqual(classify_contract("OPTSTK", "XX"), "other")
        self.assertIn("'FUTSTK'", STOCK_INDEX_FUTURES_SQL)
        self.assertIn("'OPTSTK'", STOCK_INDEX_OPTIONS_SQL)

    def test_sql_predicates_select_the_expected_raw_contract_rows(self):
        con = sqlite3.connect(":memory:")
        try:
            con.execute('CREATE TABLE contracts ("InstrumentType" TEXT, "OptionType" TEXT, "Ticker" TEXT)')
            con.executemany('INSERT INTO contracts VALUES (?, ?, ?)', [
                ("FUTCOM", "XX", "CRUDEOIL"),
                ("FUTCOM", None, "GOLD"),
                ("OPTFUT", "CE", "CRUDEOIL"),
                ("OPTFUT", "PE", "CRUDEOIL"),
                ("OPTFUT", "XX", "CRUDEOIL"),
                ("OPTCOM", "CE", "GOLD"),
                ("FUTSTK", "XX", "RELIANCE"),
                ("OPTSTK", "CE", "RELIANCE"),
            ])
            cases = [
                (COMMODITY_FUTURES_SQL, {"CRUDEOIL", "GOLD"}),
                (COMMODITY_OPTIONS_SQL, {"CRUDEOIL", "GOLD"}),
                (STOCK_INDEX_FUTURES_SQL, {"RELIANCE"}),
                (STOCK_INDEX_OPTIONS_SQL, {"RELIANCE"}),
            ]
            for predicate, expected_tickers in cases:
                actual = {row[0] for row in con.execute(f'SELECT "Ticker" FROM contracts WHERE {predicate}').fetchall()}
                self.assertEqual(actual, expected_tickers)
            self.assertEqual(
                con.execute(f'SELECT COUNT(*) FROM contracts WHERE {COMMODITY_OPTIONS_SQL}').fetchone()[0],
                3,
            )
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
