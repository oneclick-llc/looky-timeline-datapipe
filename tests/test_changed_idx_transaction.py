import os
from typing import List

import pandas as pd
import pytest
from sqlalchemy import BigInteger, Column, Integer, String, text

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


BIG_ID = 2**53 + 1  # not representable as float64


# On Postgres the changed-idx query cannot run for a non-text join key yet: the deleted-records CTE
# selects an untyped NULL for it, and UNION fails with "bigint and text cannot be matched".
@pytest.mark.skipif(os.environ.get("TEST_DB_ENV") != "sqlite", reason="needs a non-text join key, see above")
def test_bigint_join_key_survives_null_in_another_chunk(dbconn) -> None:
    ds = DataStore(dbconn, create_meta_table=True)
    users = ds.create_table(
        "users",
        TableStoreDB(
            dbconn, "users", [Column("id", BigInteger, primary_key=True), Column("name", String)], create_table=True
        ),
    )
    subs = ds.create_table(
        "subs",
        TableStoreDB(
            dbconn,
            "subs",
            [Column("id", String, primary_key=True), Column("follower_id", BigInteger)],
            create_table=True,
        ),
    )
    out = ds.create_table(
        "out",
        TableStoreDB(
            dbconn, "out", [Column("id", String, primary_key=True), Column("name", String)], create_table=True
        ),
    )

    def enrich(subs_df, followers_df):
        res = subs_df.merge(followers_df, left_on="follower_id", right_on="id", suffixes=("", "_user"))
        return res[["id", "name"]]

    step = BatchTransformStep(
        ds=ds,
        name="enrich",
        func=enrich,
        input_dts=[
            ComputeInput(dt=subs, join_type="full"),
            ComputeInput(dt=users, join_type="full", join_keys={"follower_id": "id"}),
        ],
        output_dts=[out],
        transform_keys=["id"],
        use_offset_optimization=True,
        chunk_size=1,
    )

    users.store_chunk(pd.DataFrame({"id": [BIG_ID, BIG_ID + 2], "name": ["big", "bigger"]}))
    subs.store_chunk(pd.DataFrame({"id": ["s1", "s2"], "follower_id": [BIG_ID, BIG_ID]}))
    step.run_full(ds)

    # s1 is deleted, so its follower_id comes back as NULL; s2 now follows a different big id.
    subs.delete_by_idx(pd.DataFrame({"id": ["s1"]}))
    subs.store_chunk(pd.DataFrame({"id": ["s2"], "follower_id": [BIG_ID + 2]}))

    _count, idx_gen = step.get_full_process_ids(ds)
    chunks = list(idx_gen)

    follower_ids = {row["id"]: row["follower_id"] for chunk in chunks for row in chunk.to_dict("records")}
    assert follower_ids["s2"] == BIG_ID + 2
