from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pandas as pd
import pytest
from sqlalchemy import Column, Integer, String

from datapipe.compute import ComputeInput
from datapipe.datatable import DataStore
from datapipe.step.batch_transform import BatchTransformStep
from datapipe.store.database import DBConn, TableStoreDB


def test_run_full_does_not_publish_stale_output_after_source_update(
    dbconn: DBConn,
):
    """
    A source update committed after input read must not leave a stale output.

    The transform runs on one PostgreSQL connection while a second connection
    updates the source after the input DataFrame has been read. The barrier
    makes the T1 -> T2 ordering deterministic.
    """
    if dbconn.schema is None or dbconn.connstr.startswith(("sqlite", "pysqlite")):
        pytest.skip("The race requires two independent PostgreSQL connections")

    writer_conn = DBConn(dbconn.connstr, dbconn.schema)
    ds = DataStore(dbconn, create_meta_table=True)
    writer_ds = DataStore(writer_conn, create_meta_table=False)

    input_schema = [
        Column("id", String, primary_key=True),
        Column("value", Integer),
    ]
    input_store = TableStoreDB(
        dbconn,
        "offset_race_input",
        input_schema,
        create_table=True,
    )
    input_dt = ds.create_table("offset_race_input", input_store)

    writer_input_store = TableStoreDB(
        writer_conn,
        "offset_race_input",
        [
            Column("id", String, primary_key=True),
            Column("value", Integer),
        ],
        create_table=False,
    )
    writer_input_dt = writer_ds.create_table("offset_race_input", writer_input_store)

    output_store = TableStoreDB(
        dbconn,
        "offset_race_output",
        [
            Column("id", String, primary_key=True),
            Column("value", Integer),
        ],
        create_table=True,
    )
    output_dt = ds.create_table("offset_race_output", output_store)

    source_read = Event()
    release_transform = Event()
    transform_inputs = []

    def copy_func(df: pd.DataFrame) -> pd.DataFrame:
        transform_inputs.append(df["value"].tolist())
        if len(transform_inputs) == 1:
            assert df["value"].tolist() == [1]
            source_read.set()
            if not release_transform.wait(timeout=10):
                raise AssertionError("test barrier was not released")
        return df[["id", "value"]]

    step = BatchTransformStep(
        ds=ds,
        name="offset_race_copy",
        func=copy_func,
        input_dts=[ComputeInput(dt=input_dt, join_type="full")],
        output_dts=[output_dt],
        transform_keys=["id"],
        use_offset_optimization=True,
        chunk_size=1,
    )

    input_dt.store_chunk(
        pd.DataFrame({"id": ["post-1"], "value": [1]}),
        now=100.0,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        run = executor.submit(step.run_full, ds)
        try:
            assert source_read.wait(timeout=10), "transform did not read the source"
            writer_input_dt.store_chunk(
                pd.DataFrame({"id": ["post-1"], "value": [2]}),
                now=200.0,
            )
        finally:
            release_transform.set()

        run.result(timeout=30)

    output = output_dt.get_data()
    assert transform_inputs == [[1], [2]]
    assert output[["id", "value"]].to_dict("records") == [
        {"id": "post-1", "value": 2}
    ]
    assert ds.offset_table.get_offset(step.get_name(), input_dt.name) == 200.0
