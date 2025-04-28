import pandas as pd
import numpy as np


# --- Функции для создания отдельных фичей ---

def create_datetime_features(df):
    """Создает фичи из visit_datetime."""
    if 'visit_datetime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['visit_datetime']):
        # print("Creating datetime features...") # Убрали вывод
        dt_col = df['visit_datetime']
        df['visit_hour'] = dt_col.dt.hour.astype(np.int8)
        df['visit_dayofweek'] = dt_col.dt.dayofweek.astype(np.int8)
        df['visit_month'] = dt_col.dt.month.astype(np.int8)
        df['is_weekend'] = df['visit_dayofweek'].apply(lambda x: 1 if x >= 5 else 0).astype(np.int8)
        df['is_night_hours'] = (~df['visit_hour'].between(7, 23, inclusive='left')).astype(np.int8)
    else:
        print("Warning: 'visit_datetime' column not found or not in datetime format.")
    return df


# --- Функция parse_screen_resolution УДАЛЕНА ---

def aggregate_hits_features(df_hits):
    """Агрегирует признаки из данных хитов."""
    if df_hits is None or df_hits.empty:
        print("Hits data is not available for aggregation.")
        return pd.DataFrame()

    required_hit_cols = ['session_id', 'hit_number', 'hit_time']
    optional_hit_cols = ['hit_page_path', 'hit_type']
    available_cols = [col for col in required_hit_cols + optional_hit_cols if col in df_hits.columns]

    if not all(col in available_cols for col in required_hit_cols):
        print(f"Error: Missing one or more required columns: {required_hit_cols}")
        return pd.DataFrame()

    df_hits = df_hits[available_cols].sort_values(['session_id', 'hit_number'])
    df_hits['hit_time'] = pd.to_numeric(df_hits['hit_time'], errors='coerce').fillna(0)

    session_duration = df_hits.groupby('session_id')['hit_time'].max() / 1000

    aggregations = {'hit_number': 'count'}  # n_hits
    if 'hit_page_path' in available_cols:
        aggregations['hit_page_path'] = 'nunique'  # n_unique_pages

    # Проверка на PAGE хиты (без вывода)
    has_page_hits = False
    if 'hit_type' in available_cols:
        if (df_hits['hit_type'] == 'PAGE').any():
            has_page_hits = True

    grouped_hits = df_hits.groupby('session_id').agg(aggregations)  # Используем .agg

    rename_map = {'hit_number': 'n_hits'}
    if 'hit_page_path' in aggregations:
        rename_map['hit_page_path'] = 'n_unique_pages'
    grouped_hits = grouped_hits.rename(columns=rename_map)

    grouped_hits['session_duration_sec'] = session_duration.astype(np.float32).fillna(0)

    # Новые агрегированные фичи
    grouped_hits['avg_time_per_hit_sec'] = (
                grouped_hits['session_duration_sec'] / grouped_hits['n_hits'].replace(0, 1)).astype(np.float32)

    # Флаги низкой вовлеченности
    if 'n_unique_pages' in grouped_hits.columns:
        grouped_hits['is_single_page_session'] = (grouped_hits['n_unique_pages'] <= 1).astype(np.int8)
    else:
        grouped_hits['is_single_page_session'] = 1  # Если нет инфо о страницах, считаем, что одна
    grouped_hits['is_short_session_5s'] = (grouped_hits['session_duration_sec'] < 5).astype(np.int8)

    return grouped_hits.reset_index()


def add_aggregated_features(df_sessions, df_hits):
    """Агрегирует хиты и добавляет фичи к сессиям."""
    df_hits_agg = aggregate_hits_features(df_hits)

    if not df_hits_agg.empty:
        n_sessions_before = len(df_sessions)
        df_sessions = pd.merge(df_sessions, df_hits_agg, on='session_id', how='left')
        n_sessions_after = len(df_sessions)
        if n_sessions_before != n_sessions_after:
            print(
                f"Warning: Number of sessions changed after merge! Before: {n_sessions_before}, After: {n_sessions_after}")

        agg_cols = [col for col in df_hits_agg.columns if col != 'session_id']
        fill_values = {col: 0 for col in agg_cols}
        df_sessions.fillna(fill_values, inplace=True)

        # Приводим типы данных
        for col in agg_cols:
            if 'n_hits' in col or 'is_' in col or 'n_unique_pages' in col:
                df_sessions[col] = df_sessions[col].astype(np.int32)
            elif 'duration' in col or 'avg_time' in col:
                df_sessions[col] = df_sessions[col].astype(np.float32)

    else:
        agg_cols_needed = ['n_hits', 'n_unique_pages', 'session_duration_sec',
                           'avg_time_per_hit_sec',
                           'is_single_page_session', 'is_short_session_5s']
        for col in agg_cols_needed:
            if col not in df_sessions.columns:
                default_type = np.int32 if ('n_' in col or 'is_' in col) else np.float32
                df_sessions[col] = 0
                df_sessions[col] = df_sessions[col].astype(default_type)

    return df_sessions


def create_interaction_features(df):
    """Создает комбинации признаков."""
    # Комбинация ОС и браузера
    if 'device_os' in df.columns and 'device_browser' in df.columns:
        df['device_os_browser'] = df['device_os'].astype(str) + '_' + df['device_browser'].astype(str)
        # Ограничиваем количество категорий
        top_n_comb = 30
        top_comb_idx = df['device_os_browser'].value_counts().nlargest(top_n_comb).index
        df['device_os_browser'] = df['device_os_browser'].apply(
            lambda x: x if x in top_comb_idx else 'Other_OS_Browser')
        df['device_os_browser'] = df['device_os_browser'].astype('category')

    return df


# --- Основная функция для вызова из ноутбука ---

def engineer_features(df_sessions_cleaned, df_hits=None):
    """
    Применяет все шаги фиче-инжиниринга к очищенному датасету сессий.
    (Версия после анализа EDA с CR ~3.34%)
    """
    print("\n--- Running Feature Engineering Pipeline ---")
    df_featured = df_sessions_cleaned.copy()
    df_featured = create_datetime_features(df_featured)
    df_featured = add_aggregated_features(df_featured, df_hits)
    df_featured = create_interaction_features(df_featured)

    print("--- Feature Engineering Pipeline Finished ---")
    return df_featured
