"""
🚨 КРИТИЧЕСКИЙ PRODUCTION БАГ: Offset Optimization теряет данные

ПРОБЛЕМА В PRODUCTION:
- Дата: 08.12.2025
- Потеряно: 48,915 из 82,000 записей (60%)
- Причина: Строгое неравенство в WHERE update_ts > offset

КОРНЕВАЯ ПРИЧИНА:
Код: datapipe/meta/sql_meta.py:967
  tbl.c.update_ts > offset  # ❌ ОШИБКА! Должно быть >=

МЕХАНИЗМ БАГА:
1. Батч обрабатывает записи в ORDER BY (id, hashtag) - НЕ по времени!
2. Батч содержит записи с РАЗНЫМИ update_ts
3. offset = MAX(update_ts) из обработанного батча
4. Следующий запуск: WHERE update_ts > offset (строгое неравенство!)
5. Записи с update_ts == offset но не вошедшие в батч ТЕРЯЮТСЯ

ВИЗУАЛИЗАЦИЯ ПРОБЛЕМЫ:
┌─────────────────────────────────────────────────────────────────────┐
│ Временная шкала (update_ts):                                       │
│                                                                     │
│ T1 (16:21)  T2 (17:00)  T3 (18:00)  T4 (19:00)  T5 (20:29)        │
│     │           │           │           │           │               │
│     ▼           ▼           ▼           ▼           ▼               │
│  rec_00      rec_08      rec_13      rec_18      rec_22            │
│  rec_01      rec_09      rec_14      rec_19      rec_23            │
│  ...         rec_10      rec_15      rec_20      rec_24            │
│  rec_07      rec_11      rec_16      rec_21                        │
│              rec_12      rec_17                                     │
│                                                                     │
│ ORDER BY id (сортировка для обработки):                            │
│ rec_00 → rec_01 → ... → rec_07 → rec_08 → rec_09 → rec_10 → ...   │
│   T1      T1           T1        T2        T2        T2            │
│                                                                     │
│ ┌──────────────────────────┐                                       │
│ │ БАТЧ 1 (chunk_size=10)   │                                       │
│ │ rec_00 до rec_09         │                                       │
│ │ update_ts: T1...T1, T2   │                                       │
│ └──────────────────────────┘                                       │
│           ↓                                                         │
│    offset = MAX(T1, T2) = T2                                       │
│                                                                     │
│ Следующий запуск:                                                  │
│ WHERE update_ts > T2  ← СТРОГОЕ НЕРАВЕНСТВО!                      │
│                                                                     │
│ 🚨 ПОТЕРЯНЫ:                                                       │
│    rec_10, rec_11, rec_12  (update_ts = T2 == offset)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

РЕШЕНИЕ:
Заменить > на >= в sql_meta.py:967:
  tbl.c.update_ts >= offset  # ✅ ПРАВИЛЬНО
"""

import time
from typing import List, Tuple

import pandas as pd
import pytest
from sqlalchemy import Column, Integer, String

from datapipe.compute import ComputeInput
from datapipe.datatable import DataStore
from datapipe.step.batch_transform import BatchTransformStep
from datapipe.store.database import DBConn, TableStoreDB


# ============================================================================
# ПОДГОТОВКА ТЕСТОВЫХ ДАННЫХ
# ============================================================================

def prepare_test_data() -> List[Tuple[str, str, float]]:
    """
    Подготовка тестовых данных для воспроизведения production бага.

    Данные имитируют накопление записей в течение нескольких часов
    с РАЗНЫМИ update_ts (как в реальной системе).

    Returns:
        List[(record_id, label, update_ts_offset)]

    Временная шкала:
        T1 = base_time + 1  (16:21 в production)
        T2 = base_time + 2  (17:00)
        T3 = base_time + 3  (18:00)
        T4 = base_time + 4  (19:00)
        T5 = base_time + 5  (20:29 в production)
    """
    # Формат: (id, label для логов, смещение timestamp от base_time)
    test_data = [
        # Группа 1: T1 (16:21) - 8 записей
        ("rec_00", "T1", 1.0),
        ("rec_01", "T1", 1.0),
        ("rec_02", "T1", 1.0),
        ("rec_03", "T1", 1.0),
        ("rec_04", "T1", 1.0),
        ("rec_05", "T1", 1.0),
        ("rec_06", "T1", 1.0),
        ("rec_07", "T1", 1.0),

        # Группа 2: T2 (17:00) - 5 записей
        # ⚠️ КРИТИЧНО: rec_08 и rec_09 войдут в ПЕРВЫЙ батч
        #              rec_10, rec_11, rec_12 останутся на ВТОРОЙ батч
        ("rec_08", "T2", 2.0),
        ("rec_09", "T2", 2.0),
        ("rec_10", "T2", 2.0),  # 🚨 БУДЕТ ПОТЕРЯНА
        ("rec_11", "T2", 2.0),  # 🚨 БУДЕТ ПОТЕРЯНА
        ("rec_12", "T2", 2.0),  # 🚨 БУДЕТ ПОТЕРЯНА

        # Группа 3: T3 (18:00) - 5 записей
        ("rec_13", "T3", 3.0),
        ("rec_14", "T3", 3.0),
        ("rec_15", "T3", 3.0),
        ("rec_16", "T3", 3.0),
        ("rec_17", "T3", 3.0),

        # Группа 4: T4 (19:00) - 4 записи
        ("rec_18", "T4", 4.0),
        ("rec_19", "T4", 4.0),
        ("rec_20", "T4", 4.0),
        ("rec_21", "T4", 4.0),

        # Группа 5: T5 (20:29) - 3 записи
        ("rec_22", "T5", 5.0),
        ("rec_23", "T5", 5.0),
        ("rec_24", "T5", 5.0),
    ]

    return test_data


def print_test_data_visualization(test_data: List[Tuple[str, str, float]], base_time: float):
    """Визуализация тестовых данных для отладки"""
    print("\n" + "=" * 80)
    print("ПОДГОТОВЛЕННЫЕ ТЕСТОВЫЕ ДАННЫЕ")
    print("=" * 80)
    print("\nВсего записей:", len(test_data))
    print("\nРаспределение по временным меткам:")

    # Группируем по timestamp
    by_timestamp = {}
    for record_id, label, offset in test_data:
        ts = base_time + offset
        if ts not in by_timestamp:
            by_timestamp[ts] = []
        by_timestamp[ts].append((record_id, label))

    for ts in sorted(by_timestamp.keys()):
        records = by_timestamp[ts]
        label = records[0][1]
        ids = [r[0] for r in records]
        print(f"  {label}: {len(records)} записей - {', '.join(ids)}")

    print("\nОжидаемое распределение по батчам (chunk_size=10, ORDER BY id):")
    print("  Батч 1: rec_00 до rec_09  (10 записей)")
    print("          update_ts: T1(8 записей), T2(2 записи)")
    print("          offset после батча = MAX(T1, T2) = T2")
    print()
    print("  🚨 КРИТИЧНО: Следующий запуск WHERE update_ts > T2")
    print("     ПРОПУСТИТ: rec_10, rec_11, rec_12 (update_ts == T2)")
    print()


# ============================================================================
# PRODUCTION БАГ ТЕСТ
# ============================================================================

def test_production_bug_offset_loses_records_with_equal_update_ts(dbconn: DBConn):
    """
    Тест >= неравенства в offset фильтре (исправление production бага).

    ИСХОДНЫЙ PRODUCTION БАГ (до атомарного commit):
    - 82,000 записей накоплено, chunk_size=1000
    - Обработан только ПЕРВЫЙ батч (частичная обработка)
    - offset = MAX(update_ts) из батча
    - Оставшиеся записи с update_ts == offset ПОТЕРЯНЫ (48,915 записей, 60%)
    - Причина: WHERE update_ts > offset (строгое >) вместо >=

    ПОЧЕМУ СТАРЫЙ ТЕСТ НЕ РАБОТАЕТ:
    С новым атомарным commit механизмом невозможно симулировать частичную
    обработку через run_idx() - offset коммитится только после полного run_full().

    НОВАЯ ВЕРСИЯ ТЕСТА (совместима с атомарным commit):
    1. Загружаем 25 записей с разными update_ts
    2. ПЕРВЫЙ run_full() обрабатывает ВСЕ записи, offset = MAX(update_ts)
    3. Добавляем НОВЫЕ записи с update_ts == offset (критический случай!)
    4. ВТОРОЙ run_full() должен обработать эти записи (тест >= вместо >)
    5. Проверяем что НЕТ потерь данных

    КОГДА ВОЗМОЖЕН СЦЕНАРИЙ "update_ts == offset между запусками"?
    - Clock skew между серверами (разные системные часы)
    - Backfill старых данных с прошлыми timestamp
    - Delayed records из очереди с задержкой
    - Ручное добавление записей с кастомным timestamp

    СУТЬ ТЕСТА:
    Проверяем что >= работает корректно и записи с update_ts == offset
    обрабатываются, а не теряются (независимо от сценария возникновения).
    """
    # ========== SETUP ==========
    ds = DataStore(dbconn, create_meta_table=True)

    input_store = TableStoreDB(
        dbconn,
        "production_bug_input",
        [
            Column("id", String, primary_key=True),
            Column("value", Integer),
        ],
        create_table=True,
    )
    input_dt = ds.create_table("production_bug_input", input_store)

    output_store = TableStoreDB(
        dbconn,
        "production_bug_output",
        [
            Column("id", String, primary_key=True),
            Column("value", Integer),
        ],
        create_table=True,
    )
    output_dt = ds.create_table("production_bug_output", output_store)

    def copy_func(df):
        """Простая функция копирования (как copy_to_online в production)"""
        return df[["id", "value"]]

    step = BatchTransformStep(
        ds=ds,
        name="production_bug_copy",
        func=copy_func,
        input_dts=[ComputeInput(dt=input_dt, join_type="full")],
        output_dts=[output_dt],
        transform_keys=["id"],
        use_offset_optimization=True,
        chunk_size=10,  # Маленький для быстрого теста (в production=1000)
    )

    # ========== ПОДГОТОВКА НАЧАЛЬНЫХ ДАННЫХ ==========
    base_time = time.time()
    test_data = prepare_test_data()

    # Визуализация данных
    print_test_data_visualization(test_data, base_time)

    # Загружаем начальные данные группами по timestamp
    for record_id, label, offset in test_data:
        ts = base_time + offset
        input_dt.store_chunk(
            pd.DataFrame({"id": [record_id], "value": [int(offset * 100)]}),
            now=ts
        )
        time.sleep(0.001)  # Небольшая задержка для корректности timestamp

    # Проверяем метаданные
    all_meta = input_dt.meta_table.get_metadata()
    print(f"\n✓ Всего записей загружено: {len(all_meta)}")

    # ========== ПЕРВЫЙ ЗАПУСК (обработка всех начальных записей) ==========
    print("\n" + "=" * 80)
    print("ПЕРВЫЙ ЗАПУСК ТРАНСФОРМАЦИИ (run_full)")
    print("=" * 80)

    step.run_full(ds)

    # Проверяем offset после первого запуска
    offsets = ds.offset_table.get_offsets_for_transformation(step.get_name())
    offset_after_first = offsets["production_bug_input"]

    output_after_first = output_dt.get_data()

    print(f"\n✓ Обработано записей: {len(output_after_first)}")
    print(f"✓ offset установлен на: {offset_after_first:.2f}")

    # Показываем какие записи обработаны
    processed_ids = sorted(output_after_first["id"].tolist())
    print(f"✓ Обработанные id: {', '.join(processed_ids[:5])}...{', '.join(processed_ids[-2:])}")

    # Проверяем что все начальные записи обработаны
    assert len(output_after_first) == len(test_data), (
        f"ОШИБКА: Первый run_full должен обработать все записи. "
        f"Ожидалось {len(test_data)}, получено {len(output_after_first)}"
    )

    # ========== КРИТИЧЕСКИЙ СЦЕНАРИЙ: Добавляем записи с update_ts == offset ==========
    print("\n" + "=" * 80)
    print("КРИТИЧЕСКИЙ СЦЕНАРИЙ: Добавление записей с update_ts == offset")
    print("=" * 80)

    # Добавляем НОВЫЕ записи с timestamp РАВНЫМ offset
    # Это воспроизводит production баг: записи с update_ts == offset должны обрабатываться!
    critical_timestamp = offset_after_first
    critical_records = [
        ("rec_critical_01", 999),
        ("rec_critical_02", 998),
        ("rec_critical_03", 997),
    ]

    print(f"\nДобавляем {len(critical_records)} записей с update_ts == {critical_timestamp:.2f}")
    for record_id, value in critical_records:
        input_dt.store_chunk(
            pd.DataFrame({"id": [record_id], "value": [value]}),
            now=critical_timestamp
        )
        time.sleep(0.001)

    # Проверяем что записи действительно имеют update_ts == offset
    critical_meta = input_dt.meta_table.get_metadata(
        pd.DataFrame({"id": [rec[0] for rec in critical_records]})
    )
    for idx, row in critical_meta.iterrows():
        assert abs(row["update_ts"] - critical_timestamp) < 0.01, (
            f"ОШИБКА В ТЕСТЕ: Запись {row['id']} должна иметь update_ts == offset"
        )
        print(f"  {row['id']}: update_ts={row['update_ts']:.2f} == offset={critical_timestamp:.2f}")

    # ========== ВТОРОЙ ЗАПУСК (тестируем >= вместо >) ==========
    print("\n" + "=" * 80)
    print("ВТОРОЙ ЗАПУСК ТРАНСФОРМАЦИИ (проверка >= вместо >)")
    print("=" * 80)

    # Проверяем сколько записей будет обработано
    changed_count = step.get_changed_idx_count(ds)
    print(f"Записей для обработки: {changed_count}")

    if changed_count == 0:
        pytest.fail(
            f"\n🚨 КРИТИЧЕСКИЙ БАГ В OFFSET OPTIMIZATION!\n"
            f"{'=' * 50}\n"
            f"Добавлено {len(critical_records)} НОВЫХ записей с update_ts == offset={critical_timestamp:.2f}\n"
            f"Но get_changed_idx_count вернул 0 - записи НЕ ВИДНЫ для обработки!\n\n"
            f"МЕХАНИЗМ БАГА:\n"
            f"WHERE update_ts > offset (строгое неравенство!) пропускает записи с update_ts == offset\n"
            f"Должно быть: WHERE update_ts >= offset\n\n"
            f"В PRODUCTION: Этот баг привел к потере 48,915 из 82,000 записей (60%)\n"
            f"{'=' * 50}"
        )

    # NOTE: changed_count может быть > len(critical_records) потому что >= включает
    # старые записи с update_ts == offset. Система отфильтрует их по process_ts.
    # Главное - чтобы критические записи были видны и обработаны!
    print(f"  (может включать старые записи с update_ts == offset, они будут отфильтрованы по process_ts)")

    # Запускаем обработку
    step.run_full(ds)

    # ========== ПРОВЕРКА РЕЗУЛЬТАТА ==========
    print("\n" + "=" * 80)
    print("ПРОВЕРКА РЕЗУЛЬТАТА")
    print("=" * 80)

    final_output = output_dt.get_data()
    final_processed_ids = set(final_output["id"].tolist())

    # Проверяем что все критические записи обработаны
    all_critical_processed = all(rec[0] in final_processed_ids for rec in critical_records)

    print(f"\nФинальный результат:")
    print(f"  Начальных записей:      {len(test_data)}")
    print(f"  Критических записей:    {len(critical_records)}")
    print(f"  ВСЕГО ожидается:        {len(test_data) + len(critical_records)}")
    print(f"  Обработано в output:    {len(final_output)}")

    if not all_critical_processed:
        lost_critical = [rec[0] for rec in critical_records if rec[0] not in final_processed_ids]
        print(f"\n🚨 ПОТЕРЯНЫ КРИТИЧЕСКИЕ ЗАПИСИ: {lost_critical}")

        pytest.fail(
            f"\n🚨 КРИТИЧЕСКИЙ БАГ В OFFSET OPTIMIZATION!\n"
            f"{'=' * 50}\n"
            f"Критические записи с update_ts == offset НЕ обработаны!\n"
            f"Потеряно: {len(lost_critical)} из {len(critical_records)}\n"
            f"Потерянные id: {lost_critical}\n\n"
            f"МЕХАНИЗМ БАГА:\n"
            f"WHERE update_ts > offset (строгое >) вместо >=\n"
            f"Записи с update_ts == offset пропускаются!\n\n"
            f"В PRODUCTION: 82,000 записей, потеряно 48,915 (60%)\n"
            f"{'=' * 50}"
        )

    # Финальная проверка: все записи обработаны
    expected_total = len(test_data) + len(critical_records)
    assert len(final_output) == expected_total, (
        f"Ожидалось {expected_total} записей, получено {len(final_output)}"
    )

    print(f"\n✅ Все записи обработаны корректно!")
    print(f"✅ Записи с update_ts == offset обработаны (>= работает правильно)")


if __name__ == "__main__":
    # Для ручного запуска и отладки
    from datapipe.store.database import DBConn
    from sqlalchemy import create_engine, text

    DBCONNSTR = "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
    DB_TEST_SCHEMA = "test_production_bug"

    eng = create_engine(DBCONNSTR)
    try:
        with eng.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {DB_TEST_SCHEMA} CASCADE"))
    except Exception:
        pass

    with eng.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {DB_TEST_SCHEMA}"))

    test_dbconn = DBConn(DBCONNSTR, DB_TEST_SCHEMA)

    print("Запуск теста воспроизведения production бага...")
    test_production_bug_offset_loses_records_with_equal_update_ts(test_dbconn)
