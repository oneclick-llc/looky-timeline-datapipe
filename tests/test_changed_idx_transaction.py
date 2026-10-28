import os
from typing import List

import pandas as pd
import pytest
from sqlalchemy import Column, Integer, text

from datapipe.compute import ComputeInput
from datapipe.datatable import DataStore
from datapipe.step.batch_transform import BatchTransformStep
from datapipe.store.database import TableStoreDB

TEST_SCHEMA: List[Column] = [
    Column("id", Integer, primary_key=True),
    Column("a", Integer),
]

TEST_DF = pd.DataFrame({"id": range(10), "a": range(10)})


def _idle_in_transaction(dbconn) -> int:
    with dbconn.con.connect() as con:
        return con.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND state = 'idle in transaction'"
            )
        ).scalar()


@pytest.mark.skipif(os.environ.get("TEST_DB_ENV") == "sqlite", reason="pg_stat_activity is Postgres-only")
def test_changed_idx_transaction_is_closed_while_chunks_are_processed(dbconn) -> None:
    ds = DataStore(dbconn, create_meta_table=True)
    tbl1 = ds.create_table("tbl1", table_store=TableStoreDB(dbconn, "tbl1_data", TEST_SCHEMA, True))
    tbl2 = ds.create_table("tbl2", table_store=TableStoreDB(dbconn, "tbl2_data", TEST_SCHEMA, True))
    tbl1.store_chunk(TEST_DF)

    step = BatchTransformStep(
        ds=ds,
        name="step",
        func=lambda df: df,
        input_dts=[ComputeInput(dt=tbl1, join_type="full")],
        output_dts=[tbl2],
        chunk_size=3,
    )

    count, idx_gen = step.get_full_process_ids(ds)
    try:
        chunks = [next(idx_gen)]
        # An open transaction here would also hang the fixture's DROP SCHEMA, so close the generator on failure.
        assert _idle_in_transaction(dbconn) == 0
        chunks.extend(idx_gen)
    finally:
        idx_gen.close()
    assert count == 4
    assert [len(c) for c in chunks] == [3, 3, 3, 1]
    assert sorted(pd.concat(chunks)["id"]) == list(range(10))
    assert all(list(c.index) == list(range(len(c))) for c in chunks)
