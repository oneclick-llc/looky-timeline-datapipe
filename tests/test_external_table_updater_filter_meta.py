from unittest.mock import ANY, Mock

import pandas as pd
from sqlalchemy import Column
from sqlalchemy.sql.sqltypes import Integer, String

from datapipe.compute import Catalog, Pipeline, Table, run_pipeline
from datapipe.datatable import DataStore
from datapipe.run_config import RunConfig
from datapipe.step.update_external_table import UpdateExternalTable
from datapipe.step.update_external_table import update_external_table
from datapipe.store.database import DBConn, TableStoreDB
from datapipe.tests.util import assert_df_equal


def test_external_table_updater_empty_input_still_marks_stale_rows():
    empty_df = pd.DataFrame(columns=["id"])
    stale_idx = pd.DataFrame({"id": ["stale"]})
    table = Mock()
    table.name = "test_data"
    table.table_store.read_rows_meta_pseudo_df.return_value = iter([empty_df])
    table.table_store.hash_rows.return_value = empty_df
    table.meta_table.get_changes_for_store_chunk.return_value = (
        empty_df,
        empty_df,
        empty_df,
        empty_df,
    )
    table.meta_table.get_stale_idx.return_value = iter([stale_idx])

    update_external_table(Mock(), table)

    table.meta_table.update_rows.assert_called_once()
    table.meta_table.mark_rows_deleted.assert_called_once_with(stale_idx, now=ANY)


def test_external_table_updater_filter(dbconn: DBConn):
    meta_dbconn = DBConn(dbconn.connstr, dbconn.schema)
    test_store = TableStoreDB(
        dbconn=dbconn,
        name="test_data",
        data_sql_schema=[
            Column("composite_id_1", Integer(), primary_key=True),
            Column("composite_id_2", Integer(), primary_key=True),
            Column("data", String()),
        ],
        create_table=True,
    )
    df_test = pd.DataFrame(
        {
            "composite_id_1": [1, 1, 2, 2],
            "composite_id_2": [3, 4, 5, 6],
            "data": ["a", "b", "c", "d"],
        }
    )

    catalog = Catalog(
        {
            "test": Table(store=test_store),
        }
    )
    pipeline = Pipeline([UpdateExternalTable(output="test")])
    ds = DataStore(meta_dbconn, create_meta_table=True)

    test_store.insert_rows(df_test)

    run_pipeline(ds, catalog, pipeline)
    assert_df_equal(
        catalog.get_datatable(ds, "test").get_data(),
        df_test,
        index_cols=["composite_id_1", "composite_id_2"],
    )

    config = RunConfig(filters={"composite_id_1": 2})
    run_pipeline(ds, catalog, pipeline, run_config=config)
    assert_df_equal(
        catalog.get_datatable(ds, "test").get_data(),
        df_test,
        index_cols=["composite_id_1", "composite_id_2"],
    )
