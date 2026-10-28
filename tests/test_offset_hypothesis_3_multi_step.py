"""
Тест для гипотезы 3: Рассинхронизация update_ts и process_ts в multi-step pipeline.

ГИПОТЕЗА 3 (из README.md):
"Другая трансформация обновляет process_ts, но НЕ update_ts"

СЦЕНАРИЙ ИЗ PRODUCTION:
1. Transform_extract_hashtags создает записи в hashtag_table (update_ts=16:21)
2. Transform_hashtag_stats обрабатывает hashtag_table спустя 4 часа (20:04)
   - process_ts в Transform_hashtag_stats.meta_table = 20:04
   - update_ts в hashtag_table (input) остается = 16:21
3. offset(Transform_hashtag_stats, hashtag_table) = 16:21 (MAX update_ts из input)
4. Временной разрыв: update_ts=16:21, process_ts=20:04

ВОПРОС:
Влияет ли эта рассинхронизация на offset optimization?
Может ли это вызвать потерю данных?

АРХИТЕКТУРА:
- У каждой трансформации СВОЯ TransformMetaTable с СВОИМ process_ts
- Transform_A.meta_table хранит process_ts для Transform_A
- Transform_B.meta_table хранит process_ts для Transform_B
- Они не пересекаются!

ОЖИДАНИЕ:
Рассинхронизация НЕ должна влиять на корректность offset optimization,
так как каждая трансформация работает со своим process_ts.
"""
import time

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy import Column, Integer, String

from datapipe.compute import ComputeInput
from datapipe.datatable import DataStore
from datapipe.step.batch_transform import BatchTransformStep
from datapipe.store.database import DBConn, TableStoreDB


def test_hypothesis_3_multi_step_pipeline_update_ts_vs_process_ts_desync(dbconn: DBConn):
    """
    Проверка гипотезы 3: Рассинхронизация update_ts и process_ts в multi-step pipeline.

    Pipeline:
    source_table → Transform_A → intermediate_table → Transform_B → final_table

    Сценарий:
    1. В T1 (16:21): Transform_A создает данные в intermediate_table (update_ts=T1)
    2. В T2 (20:04): Transform_B обрабатывает intermediate_table
       - process_ts в Transform_B.meta = T2
       - update_ts в intermediate_table остается = T1
       - offset(Transform_B, intermediate_table) = T1
    3. Добавляем новые данные в source_table
    4. В T3: Transform_A обрабатывает новые данные (update_ts=T3 в intermediate)
    5. В T4: Transform_B обрабатывает новые данные
       - Проверяем что offset optimization работает корректно
       - Проверяем что старые данные не обрабатываются повторно
    """
    ds = DataStore(dbconn, create_meta_table=True)

    # ========== СОЗДАНИЕ ТАБЛИЦ ==========
    # Source table
    source_store = TableStoreDB(
        dbconn,
        "hyp3_source",
        [Column("id", String, primary_key=True), Column("value", Integer)],
        create_table=True,
    )
    source_table = ds.create_table("hyp3_source", source_store)

    # Intermediate table (output Transform_A, input Transform_B)
    intermediate_store = TableStoreDB(
        dbconn,
        "hyp3_intermediate",
        [Column("id", String, primary_key=True), Column("value_doubled", Integer)],
        create_table=True,
    )
    intermediate_table = ds.create_table("hyp3_intermediate", intermediate_store)

    # Final table (output Transform_B)
    final_store = TableStoreDB(
        dbconn,
        "hyp3_final",
        [Column("id", String, primary_key=True), Column("value_squared", Integer)],
        create_table=True,
    )
    final_table = ds.create_table("hyp3_final", final_store)

    # ========== СОЗДАНИЕ ТРАНСФОРМАЦИЙ ==========
    def double_func(df):
        """Transform_A: удваивает значения"""
        return df.assign(value_doubled=df["value"] * 2)[["id", "value_doubled"]]

    transform_a = BatchTransformStep(
        ds=ds,
        name="transform_a_double",
        func=double_func,
        input_dts=[ComputeInput(dt=source_table, join_type="full")],
        output_dts=[intermediate_table],
        transform_keys=["id"],
        use_offset_optimization=True,
        chunk_size=10,
    )

    def square_func(df):
        """Transform_B: возводит в квадрат"""
        return df.assign(value_squared=df["value_doubled"] ** 2)[["id", "value_squared"]]

    transform_b = BatchTransformStep(
        ds=ds,
        name="transform_b_square",
        func=square_func,
        input_dts=[ComputeInput(dt=intermediate_table, join_type="full")],
        output_dts=[final_table],
        transform_keys=["id"],
        use_offset_optimization=True,
        chunk_size=10,
    )

    # ========== ФАЗА 1: T1 (16:21) - Transform_A создает данные ==========
    print("\n" + "=" * 80)
    print("ФАЗА 1: T1 (16:21) - Transform_A создает первую партию данных")
    print("=" * 80)

    base_time = time.time()
    t1 = base_time + 1  # 16:21 в production

    # Загружаем данные в source
    source_data_1 = pd.DataFrame({
        "id": [f"rec_{i:02d}" for i in range(5)],
        "value": [1, 2, 3, 4, 5],
    })
    source_table.store_chunk(source_data_1, now=t1)
    time.sleep(0.01)

    # Transform_A обрабатывает source → intermediate
    transform_a.run_full(ds=ds, run_config=None)

    # Проверяем результат
    intermediate_data = intermediate_table.get_data()
    intermediate_meta = intermediate_table.meta_table.get_metadata()

    print(f"\nIntermediate table после Transform_A:")
    print(f"  Записей: {len(intermediate_data)}")
    print(f"  update_ts: {intermediate_meta['update_ts'].unique()}")

    # Сохраняем update_ts первой партии
    intermediate_update_ts_phase1 = intermediate_meta['update_ts'].iloc[0]

    assert len(intermediate_data) == 5
    assert (intermediate_data["value_doubled"] == [2, 4, 6, 8, 10]).all()

    # ========== ФАЗА 2: T2 (20:04) - Transform_B обрабатывает (С ЗАДЕРЖКОЙ!) ==========
    print("\n" + "=" * 80)
    print("ФАЗА 2: T2 (20:04) - Transform_B обрабатывает intermediate (4 часа спустя!)")
    print("=" * 80)

    # Симулируем задержку 4 часа
    time.sleep(0.05)  # В тесте - маленькая задержка
    t2 = base_time + 2  # 20:04 в production

    # Запоминаем текущее время перед обработкой
    before_transform_b = time.time()

    # Transform_B обрабатывает intermediate → final
    transform_b.run_full(ds=ds, run_config=None)

    after_transform_b = time.time()

    # Проверяем результат
    final_data = final_table.get_data()

    # Получаем данные из Transform_B meta table через SQL
    with ds.meta_dbconn.con.begin() as con:
        transform_b_meta = pd.read_sql(
            sa.select(transform_b.meta_table.sql_table),
            con
        )

    # КРИТИЧНО: проверяем update_ts в intermediate table - он НЕ должен измениться!
    intermediate_meta_after_b = intermediate_table.meta_table.get_metadata()

    print(f"\nIntermediate table после Transform_B:")
    print(f"  update_ts: {intermediate_meta_after_b['update_ts'].unique()}")
    print(f"  (должен остаться = T1, потому что Transform_B читает но НЕ пишет в intermediate)")

    print(f"\nTransform_B meta table:")
    print(f"  process_ts: {transform_b_meta['process_ts'].unique()}")
    print(f"  (должен быть ≈ T2, текущее время обработки)")

    # Проверяем offset
    offsets_b = ds.offset_table.get_offsets_for_transformation(transform_b.get_name())
    offset_b_intermediate = offsets_b["hyp3_intermediate"]

    print(f"\nOffset для Transform_B:")
    print(f"  offset(Transform_B, intermediate_table) = {offset_b_intermediate:.2f}")
    print(f"  (должен быть = MAX(update_ts из intermediate) ≈ T1)")

    print(f"\n🔍 РАССИНХРОНИЗАЦИЯ:")
    print(f"  update_ts в intermediate = {intermediate_update_ts_phase1:.2f} (T1)")
    print(f"  process_ts в Transform_B = {transform_b_meta['process_ts'].iloc[0]:.2f} (T2)")
    print(f"  Разница: {transform_b_meta['process_ts'].iloc[0] - intermediate_update_ts_phase1:.2f} сек")

    # ПРОВЕРКИ
    assert len(final_data) == 5, f"Должно быть 5 записей, получено {len(final_data)}"

    # КРИТИЧНО: Проверяем что ВСЕ записи обработаны (ничего не потеряно!)
    expected_ids_phase2 = {f"rec_{i:02d}" for i in range(5)}
    actual_ids_phase2 = set(final_data["id"].tolist())
    lost_ids_phase2 = expected_ids_phase2 - actual_ids_phase2

    if lost_ids_phase2:
        pytest.fail(
            f"🚨 ПОТЕРЯ ДАННЫХ В ФАЗЕ 2!\n"
            f"Ожидалось: {len(expected_ids_phase2)} записей\n"
            f"Обработано: {len(actual_ids_phase2)} записей\n"
            f"ПОТЕРЯНО: {len(lost_ids_phase2)} записей: {sorted(lost_ids_phase2)}\n"
            f"Это означает что гипотеза 3 влияет на потерю данных!"
        )

    print(f"\n✓ ВСЕ записи обработаны: {len(actual_ids_phase2)}/{len(expected_ids_phase2)}")

    # Проверяем значения
    assert (final_data["value_squared"] == [4, 16, 36, 64, 100]).all()

    # update_ts в intermediate НЕ должен измениться (Transform_B только читает)
    assert (intermediate_meta_after_b['update_ts'] == intermediate_update_ts_phase1).all(), \
        "update_ts в intermediate table НЕ должен измениться после Transform_B"

    # process_ts в Transform_B должен быть ≈ текущее время
    assert all(before_transform_b <= ts <= after_transform_b
               for ts in transform_b_meta['process_ts']), \
        "process_ts в Transform_B должен быть текущим временем обработки"

    # offset должен быть = MAX(update_ts из intermediate) ≈ T1
    assert abs(offset_b_intermediate - intermediate_update_ts_phase1) < 0.1, \
        "offset должен быть равен MAX(update_ts из intermediate)"

    # ========== ФАЗА 3: T3 - Добавляем новые данные через Transform_A ==========
    print("\n" + "=" * 80)
    print("ФАЗА 3: T3 - Добавляем новые данные и обрабатываем через Transform_A")
    print("=" * 80)

    time.sleep(0.01)
    t3 = base_time + 3

    # Добавляем новые данные в source
    source_data_2 = pd.DataFrame({
        "id": [f"rec_{i:02d}" for i in range(5, 10)],
        "value": [6, 7, 8, 9, 10],
    })
    source_table.store_chunk(source_data_2, now=t3)
    time.sleep(0.01)

    # Transform_A обрабатывает новые данные
    transform_a.run_full(ds=ds, run_config=None)

    intermediate_data_after_new = intermediate_table.get_data()
    print(f"\nIntermediate table после добавления новых данных:")
    print(f"  Записей: {len(intermediate_data_after_new)}")

    assert len(intermediate_data_after_new) == 10

    # ========== ФАЗА 4: T4 - Transform_B обрабатывает новые данные ==========
    print("\n" + "=" * 80)
    print("ФАЗА 4: T4 - Transform_B обрабатывает ТОЛЬКО новые данные")
    print("=" * 80)

    time.sleep(0.01)
    t4 = base_time + 4

    # Получаем количество записей в transform_b.meta ДО обработки
    with ds.meta_dbconn.con.begin() as con:
        transform_b_meta_before = pd.read_sql(
            sa.select(transform_b.meta_table.sql_table),
            con
        )
    process_ts_before = dict(zip(transform_b_meta_before['id'], transform_b_meta_before['process_ts']))

    print(f"\nTransform_B meta ДО обработки новых данных:")
    print(f"  Записей: {len(transform_b_meta_before)}")
    print(f"  process_ts для старых записей (rec_00): {process_ts_before.get('rec_00', 'N/A')}")

    # Transform_B обрабатывает новые данные с использованием offset
    transform_b.run_full(ds=ds, run_config=None)

    final_data_after_new = final_table.get_data()
    with ds.meta_dbconn.con.begin() as con:
        transform_b_meta_after = pd.read_sql(
            sa.select(transform_b.meta_table.sql_table),
            con
        )
    process_ts_after = dict(zip(transform_b_meta_after['id'], transform_b_meta_after['process_ts']))

    print(f"\nTransform_B meta ПОСЛЕ обработки новых данных:")
    print(f"  Записей: {len(transform_b_meta_after)}")
    print(f"  process_ts для старых записей (rec_00): {process_ts_after.get('rec_00', 'N/A')}")
    print(f"  process_ts для новых записей (rec_05): {process_ts_after.get('rec_05', 'N/A')}")

    # КРИТИЧНАЯ ПРОВЕРКА: старые записи НЕ должны обработаться повторно
    old_ids = [f"rec_{i:02d}" for i in range(5)]
    reprocessed_ids = []

    for old_id in old_ids:
        if old_id in process_ts_before and old_id in process_ts_after:
            if abs(process_ts_after[old_id] - process_ts_before[old_id]) > 0.001:
                reprocessed_ids.append(old_id)

    if reprocessed_ids:
        print(f"\n🚨 ПРОБЛЕМА: Старые записи обработаны ПОВТОРНО!")
        print(f"  Записи: {reprocessed_ids}")
        for rid in reprocessed_ids:
            print(f"    {rid}: process_ts ДО={process_ts_before[rid]:.6f}, "
                  f"ПОСЛЕ={process_ts_after[rid]:.6f}")
        pytest.fail(
            f"ГИПОТЕЗА 3: Рассинхронизация update_ts и process_ts вызвала повторную обработку!\n"
            f"Старые записи ({reprocessed_ids}) были обработаны повторно.\n"
            f"Это указывает на проблему с offset optimization в multi-step pipeline."
        )

    # КРИТИЧНО: Проверяем что ВСЕ записи обработаны (ничего не потеряно!)
    expected_ids_phase4 = {f"rec_{i:02d}" for i in range(10)}  # rec_00..rec_09
    actual_ids_phase4 = set(final_data_after_new["id"].tolist())
    lost_ids_phase4 = expected_ids_phase4 - actual_ids_phase4

    if lost_ids_phase4:
        pytest.fail(
            f"🚨 ПОТЕРЯ ДАННЫХ В ФАЗЕ 4!\n"
            f"Ожидалось: {len(expected_ids_phase4)} записей\n"
            f"Обработано: {len(actual_ids_phase4)} записей\n"
            f"ПОТЕРЯНО: {len(lost_ids_phase4)} записей: {sorted(lost_ids_phase4)}\n\n"
            f"Детали:\n"
            f"  Старые записи (должны остаться): rec_00..rec_04\n"
            f"  Новые записи (должны добавиться): rec_05..rec_09\n"
            f"  Фактически потерянные: {sorted(lost_ids_phase4)}\n\n"
            f"Это означает что гипотеза 3 (рассинхронизация update_ts/process_ts)\n"
            f"влияет на потерю данных в multi-step pipeline!"
        )

    print(f"\n✓ ВСЕ записи обработаны: {len(actual_ids_phase4)}/{len(expected_ids_phase4)}")
    print(f"  Старые записи (rec_00..rec_04): {all(f'rec_{i:02d}' in actual_ids_phase4 for i in range(5))}")
    print(f"  Новые записи (rec_05..rec_09): {all(f'rec_{i:02d}' in actual_ids_phase4 for i in range(5, 10))}")

    # ПРОВЕРКИ
    assert len(final_data_after_new) == 10, f"Должно быть 10 записей, получено {len(final_data_after_new)}"

    # Проверяем что все значения корректны
    expected_values = [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]  # (value*2)^2
    actual_values_sorted = sorted(final_data_after_new["value_squared"].tolist())
    assert actual_values_sorted == expected_values, \
        f"Значения не совпадают. Ожидалось: {expected_values}, получено: {actual_values_sorted}"

    print(f"\n✅ ГИПОТЕЗА 3: Рассинхронизация update_ts и process_ts НЕ вызывает проблем")
    print(f"✓ Старые записи НЕ обработаны повторно")
    print(f"✓ Новые записи обработаны корректно")
    print(f"✓ Все записи обработаны, НИЧЕГО не потеряно")
    print(f"✓ offset optimization работает корректно в multi-step pipeline")
    print(f"\nОбъяснение:")
    print(f"  - У каждой трансформации СВОЯ meta table с СВОИМ process_ts")
    print(f"  - Transform_B использует offset на основе update_ts из intermediate table")
    print(f"  - process_ts в Transform_B.meta НЕ связан с update_ts в intermediate")
    print(f"  - Рассинхронизация НЕ влияет на корректность offset optimization")


def test_hypothesis_3_explanation():
    """
    Документация и объяснение гипотезы 3.

    ВОПРОС: Почему рассинхронизация update_ts и process_ts не влияет на offset optimization?

    ОТВЕТ:
    1. У каждой трансформации СВОЯ TransformMetaTable с СВОИМ process_ts
    2. offset(Transform_B, TableA) = MAX(update_ts из TableA)
    3. process_ts в Transform_B.meta относится к обработке Transform_B
    4. Они не пересекаются!

    АРХИТЕКТУРА:

    source_table → Transform_A → intermediate_table → Transform_B → final_table
                    [Meta_A]                            [Meta_B]

    - Meta_A.process_ts = когда Transform_A обработал записи
    - intermediate_table.update_ts = когда Transform_A записал данные
    - offset(Transform_B, intermediate) = MAX(intermediate_table.update_ts)
    - Meta_B.process_ts = когда Transform_B обработал записи

    ВАЖНО:
    - Transform_B НЕ смотрит на Meta_A.process_ts
    - Transform_B использует intermediate_table.update_ts для offset
    - Рассинхронизация между Meta_A.process_ts и intermediate_table.update_ts не важна

    КОГДА НУЖНА ПРОВЕРКА process_ts:
    Проверка process_ts нужна для ОДНОЙ трансформации, чтобы не обработать
    одни и те же данные дважды при изменении > на >=.

    Но это проверка СВОЕГО process_ts (Transform_B.meta.process_ts),
    а не process_ts других трансформаций!
    """
    pass


if __name__ == "__main__":
    from datapipe.store.database import DBConn
    from sqlalchemy import create_engine, text

    DBCONNSTR = "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
    DB_TEST_SCHEMA = "test_hypothesis_3_multi_step"

    eng = create_engine(DBCONNSTR)
    try:
        with eng.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {DB_TEST_SCHEMA} CASCADE"))
    except Exception:
        pass

    with eng.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {DB_TEST_SCHEMA}"))

    test_dbconn = DBConn(DBCONNSTR, DB_TEST_SCHEMA)

    print("Запуск теста гипотезы 3 (multi-step pipeline)...")
    test_hypothesis_3_multi_step_pipeline_update_ts_vs_process_ts_desync(test_dbconn)
