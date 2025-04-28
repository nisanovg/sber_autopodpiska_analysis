# src/data_processing.py

import pandas as pd
import numpy as np
import gc
import warnings

# --- Константы ---
TARGET_ACTIONS = [
    # Заявки / Отправка форм
    'sub_submit_success',
    'sub_car_claim_submit_click',
    'sub_car_request_submit_click',
    'sub_callback_submit_click',
    'form_request_call_sent',
    'request_success',
    'greenday_sub_submit_success',
    'greenday_sub_callback_submit_click',
    'offline message sent',

    # Заказ звонка / Клик по номеру
    'callback requested',
    'sub_call_number_click',
    'tap_on_phone_800',
    'tap_on_phone_495',
    'mobile call',
    'greenday_sub_call_number_click',

    # Диалог / Чат
    'sub_open_dialog_click',
    'start_chat',
    'client initiate chat',
    'proactive invitation accepted',
    'user gave contacts during chat',
    'greenday_sub_open_dialog_click',
]

# Список utm_medium для органического трафика (согласно глоссарию + обработка (none))
ORGANIC_MEDIUMS = ['organic', 'referral', '(none)']

# Список utm_medium для социального трафика (приблизительный)
SOCIAL_MEDIUMS = ['social', 'smm', 'social_media', 'stories', 'blogger_channel', 'blogger_stories']

# Значения, которые нужно заменить при очистке категориальных признаков
REPLACE_VALUES = ['', '(not set)', '(none)', '(not provided)', 'unknown.unknown', '<NA>', 'nan', 'Nan', 'NaN']
NAN_REPLACEMENT = 'unknown'  # На что заменять пропуски и нежелательные значения


# --- Функции ---

def load_data(sessions_path, hits_path=None, load_hits=False):
    """
    Загружает датасеты сессий и (опционально) хитов из pkl файлов.
    """
    print(f"Loading sessions data from: {sessions_path}")
    try:
        df_sessions = pd.read_pickle(sessions_path)
        print(f"-> Sessions loaded successfully: {df_sessions.shape}")
    except Exception as e:
        print(f"ERROR loading sessions file: {e}")
        return None, None

    df_hits = None
    if load_hits and hits_path:
        print(f"Loading hits data from: {hits_path} (minimal columns)")
        hits_cols_to_load = ['session_id', 'hit_number', 'hit_time',
                             'hit_page_path', 'event_action', 'hit_type']
        try:
            df_hits_full = pd.read_pickle(hits_path)
            hits_cols_exist = [col for col in hits_cols_to_load if col in df_hits_full.columns]
            if not hits_cols_exist:
                print("ERROR: None of the required hits columns found in the file.")
                return df_sessions, None

            df_hits = df_hits_full[hits_cols_exist]
            del df_hits_full
            gc.collect()
            print(f"-> Hits loaded and filtered successfully: {df_hits.shape}")
        except MemoryError:
            print("ERROR: Memory Error loading hits file. Data is too large.")
            warnings.warn("Could not load hits data due to MemoryError.", ResourceWarning)
            return df_sessions, None
        except Exception as e:
            print(f"ERROR loading hits file: {e}")
            return df_sessions, None

    return df_sessions, df_hits


def define_target(df_sessions, df_hits):
    """
    Определяет целевую переменную 'target' в df_sessions на основе TARGET_ACTIONS в df_hits.
    """
    if df_hits is None or df_hits.empty or 'event_action' not in df_hits.columns:
        print("WARNING: Hits data or 'event_action' column is not available. Setting target to 0 for all sessions.")
        df_sessions['target'] = 0
        return df_sessions

    print(f"Defining target based on {len(TARGET_ACTIONS)} event_action(s)...")
    valid_event_actions = df_hits['event_action'].dropna()
    target_sessions = set(df_hits.loc[valid_event_actions.isin(TARGET_ACTIONS), 'session_id'].unique())

    n_target_sessions = len(target_sessions)
    print(f"Found {n_target_sessions} sessions with at least one target action.")
    if n_target_sessions == 0:
        warnings.warn("No sessions found with the specified target actions! Check TARGET_ACTIONS list.", UserWarning)

    df_sessions['target'] = df_sessions['session_id'].apply(lambda x: 1 if x in target_sessions else 0)

    target_distribution = df_sessions['target'].value_counts(normalize=True)
    print("Target variable distribution:")
    print(target_distribution.round(4))
    # Предупреждение о сильном дисбалансе (если он все еще есть)
    positive_ratio = target_distribution.get(1, 0)
    if positive_ratio < 0.01 or positive_ratio > 0.99:
        warnings.warn(f"Target class distribution is highly imbalanced: {positive_ratio:.2%} positive class.",
                      UserWarning)
    elif positive_ratio < 0.05 or positive_ratio > 0.95:
        print(f"Note: Target class distribution is imbalanced: {positive_ratio:.2%} positive class.")

    return df_sessions


def basic_cleaning(df):
    """
    Выполняет базовую очистку DataFrame сессий.
    """
    print("Starting basic data cleaning...")
    df_clean = df.copy()

    # 1. Обработка даты и времени
    if 'visit_date' in df_clean.columns and 'visit_time' in df_clean.columns:
        try:
            date_str = df_clean['visit_date'].astype(str)
            time_str = df_clean['visit_time'].astype(str)
            datetime_str = date_str + ' ' + time_str
            df_clean['visit_datetime'] = pd.to_datetime(datetime_str, errors='coerce')
            df_clean = df_clean.drop(['visit_date', 'visit_time'], axis=1)
        except Exception as e:
            print(f"WARNING: Error creating 'visit_datetime': {e}. Original columns kept.")

    # 2. Обработка пропусков и констант в категориальных колонках
    categorical_cols_to_clean = [
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent ', 'utm_keyword',
        'device_category', 'device_os', 'device_brand', 'device_model',
        'device_browser', 'geo_country', 'geo_city'
    ]

    for col in categorical_cols_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).fillna(NAN_REPLACEMENT)
            df_clean[col] = df_clean[col].replace(REPLACE_VALUES, NAN_REPLACEMENT)

    # 3. Обработка device_screen_resolution (только заполнение NaN)
    if 'device_screen_resolution' in df_clean.columns:
        df_clean['device_screen_resolution'] = df_clean['device_screen_resolution'].fillna('0x0')

    # 4. Типизация данных
    for col in categorical_cols_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    if 'device_category' in df_clean.columns and 'device_category' not in categorical_cols_to_clean:
        df_clean['device_category'] = df_clean['device_category'].astype(str).fillna(NAN_REPLACEMENT).replace(
            REPLACE_VALUES, NAN_REPLACEMENT).astype('category')

    if 'visit_number' in df_clean.columns:
        df_clean['visit_number'] = pd.to_numeric(df_clean['visit_number'], errors='coerce').fillna(0).astype(np.int32)

    # 5. Удаление дубликатов сессий
    if 'session_id' in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['session_id'])

    print(f"Basic cleaning finished. Resulting shape: {df_clean.shape}")
    return df_clean


def classify_traffic(df):
    """
    Классифицирует типы трафика ('organic', 'paid', 'social', 'other')
    на основе utm_medium.
    """
    print("Classifying traffic types based on 'utm_medium'...")

    if 'utm_medium' not in df.columns:
        print("WARNING: 'utm_medium' column not found. Assigning 'other'.")
        df['traffic_type'] = 'other'
        df['traffic_type'] = df['traffic_type'].astype('category')
        return df

    utm_medium_lower = df['utm_medium'].astype(str).str.lower()
    is_organic = utm_medium_lower.isin([m.lower() for m in ORGANIC_MEDIUMS])
    is_social = utm_medium_lower.isin([m.lower() for m in SOCIAL_MEDIUMS])

    df['traffic_type'] = 'other'
    df.loc[is_organic, 'traffic_type'] = 'organic'
    df.loc[is_social, 'traffic_type'] = 'social'
    df.loc[~(is_organic | is_social) & (df['traffic_type'] == 'other'), 'traffic_type'] = 'paid'
    df['traffic_type'] = df['traffic_type'].astype('category')

    print("Traffic types classified. Distribution:")
    print(df['traffic_type'].value_counts(normalize=True).round(3))

    return df
