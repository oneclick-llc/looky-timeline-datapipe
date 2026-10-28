"""
A non-text additional join key (here a BIGINT foreign key) must not break the changed-idx query.

Branches of the changed-idx UNION that have no such column select NULL for it. An untyped NULL in a CTE
is resolved as text on Postgres, and the UNION failed with "UNION types bigint and text cannot be
matched" for every run of the step, not only when something was deleted.
"""

import pandas as pd
from sqlalchemy import BigInteger, Column, String

from datapipe.compute import ComputeInput
from datapipe.datatable import DataStore
from datapipe.step.batch_transform import BatchTransformStep
from datapipe.store.database import TableStoreDB

BIG_ID = 2**53 + 1  # not representable as float64


def _step(dbconn):
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
            dbconn,
            "out",
            [Column("id", String, primary_key=True), Column("name", String)],
            create_table=True,
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
    return ds, step, users, subs, out


def _processed_ids(ds, step):
    _count, idx_gen = step.get_full_process_ids(ds)
    return {row["id"]: row["follower_id"] for chunk in idx_gen for row in chunk.to_dict("records")}


def test_bigint_join_key_with_deleted_record(dbconn) -> None:
    ds, step, users, subs, out = _step(dbconn)
    users.store_chunk(pd.DataFrame({"id": [BIG_ID, BIG_ID + 2], "name": ["big", "bigger"]}))
    subs.store_chunk(pd.DataFrame({"id": ["s1", "s2"], "follower_id": [BIG_ID, BIG_ID]}))

    assert _processed_ids(ds, step) == {"s1": BIG_ID, "s2": BIG_ID}
    step.run_full(ds)
    assert sorted(out.get_data()["name"]) == ["big", "big"]

    subs.delete_by_idx(pd.DataFrame({"id": ["s1"]}))
    subs.store_chunk(pd.DataFrame({"id": ["s2"], "follower_id": [BIG_ID + 2]}))

    assert _processed_ids(ds, step)["s2"] == BIG_ID + 2
    step.run_full(ds)
    assert out.get_data()[["id", "name"]].to_dict("records") == [{"id": "s2", "name": "bigger"}]


def test_bigint_join_key_with_changed_reference(dbconn) -> None:
    ds, step, users, subs, out = _step(dbconn)
    users.store_chunk(pd.DataFrame({"id": [BIG_ID], "name": ["big"]}))
    subs.store_chunk(pd.DataFrame({"id": ["s1"], "follower_id": [BIG_ID]}))
    step.run_full(ds)

    users.store_chunk(pd.DataFrame({"id": [BIG_ID], "name": ["renamed"]}))

    step.run_full(ds)
    assert out.get_data()[["id", "name"]].to_dict("records") == [{"id": "s1", "name": "renamed"}]
