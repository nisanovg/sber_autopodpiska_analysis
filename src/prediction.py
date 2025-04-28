import pandas as pd
import joblib
import os
import numpy as np
import traceback

# Пути к сохраненным артефактам (относительно папки src)
DEFAULT_MODELS_DIR = '../models'
DEFAULT_PREPROCESSOR_FILENAME = 'preprocessor_final.joblib'  # Используем финальное имя
DEFAULT_MODEL_FILENAME = 'model_final.joblib'  # Используем финальное имя

# Глобальные переменные для хранения загруженных объектов
preprocessor = None
model = None
required_columns = None  # Список колонок, которые ожидает препроцессор


def load_prediction_assets(
        models_dir=DEFAULT_MODELS_DIR,
        preprocessor_filename=DEFAULT_PREPROCESSOR_FILENAME,
        model_filename=DEFAULT_MODEL_FILENAME
):
    """Загружает препроцессор и модель для предсказаний."""
    global preprocessor, model, required_columns

    base_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessor_path_abs = os.path.join(base_dir, models_dir, preprocessor_filename)
    model_path_abs = os.path.join(base_dir, models_dir, model_filename)

    print(f"Attempting to load preprocessor from: {preprocessor_path_abs}")
    if not os.path.exists(preprocessor_path_abs):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path_abs}")

    print(f"Attempting to load model from: {model_path_abs}")
    if not os.path.exists(model_path_abs):
        raise FileNotFoundError(f"Model file not found at {model_path_abs}")

    try:
        print("Loading preprocessor...")
        preprocessor = joblib.load(preprocessor_path_abs)
        print("Loading model...")
        model = joblib.load(model_path_abs)
        print("Prediction assets loaded successfully.")
    except Exception as e:
        print(f"Error loading prediction assets: {e}")
        traceback.print_exc()
        raise e

    # Получаем список колонок, которые ожидает препроцессор
    try:
        if hasattr(preprocessor, 'feature_names_in_'):
            required_columns = list(preprocessor.feature_names_in_)
            print(f"Preprocessor expects {len(required_columns)} columns.")
        elif hasattr(preprocessor, 'transformers_'):
            cols = []
            if hasattr(preprocessor, 'get_feature_names_out'):
                for name, _, features in preprocessor.transformers_:
                    if isinstance(features, (list, np.ndarray)) and all(
                            isinstance(f, str) for f in features): cols.extend(features)
                required_columns = list(dict.fromkeys(cols))
            else:
                for name, _, features in preprocessor.transformers_:
                    if isinstance(features, (list, np.ndarray)) and all(
                            isinstance(f, str) for f in features): cols.extend(features)
                required_columns = list(dict.fromkeys(cols))
            if required_columns:
                print(f"Inferred required columns from transformers: {len(required_columns)} columns.")
            else:
                raise ValueError("Could not determine required columns from preprocessor transformers.")
        else:
            raise ValueError("Preprocessor structure not recognized to determine required columns.")
    except Exception as e:
        print(f"Warning: Could not automatically determine required columns: {e}")
        required_columns = None


def predict_single(input_data: dict) -> tuple[int, float]:
    """
    Делает предсказание для одного объекта (словаря).
    """
    global preprocessor, model, required_columns
    if preprocessor is None or model is None:
        raise RuntimeError("Prediction assets not loaded. Call load_prediction_assets() first.")

    print(f"Received input data keys: {list(input_data.keys())}")
    try:
        df = pd.DataFrame([input_data])
        if required_columns:
            print(f"Verifying and ordering columns based on preprocessor expectations...")
            input_keys = set(df.columns)
            required_keys = set(required_columns)
            missing_keys = required_keys - input_keys
            if missing_keys:
                print(f"Warning: Missing required columns: {missing_keys}. Filling with NaN.")
                for key in missing_keys: df[key] = np.nan
            extra_keys = input_keys - required_keys
            if extra_keys: print(f"Warning: Extra columns found: {extra_keys}. They will be ignored.")
            try:
                df = df[required_columns]
            except KeyError as e:
                raise ValueError(f"Error selecting required columns: {e}.")
        else:
            print("Warning: Required columns order unknown.")
    except Exception as e:
        print(f"Error preparing input DataFrame: {e}")
        traceback.print_exc()
        raise ValueError(f"Error preparing input data: {e}")

    try:
        print("Applying preprocessor...")
        data_processed = preprocessor.transform(df)
        print(f"Data shape after preprocessing: {data_processed.shape}")
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        print("Input data passed to preprocessor:\n", df.head().to_string())
        print("Input data dtypes:\n", df.dtypes)
        traceback.print_exc()
        raise RuntimeError(f"Error during preprocessing: {e}")

    try:
        print("Making prediction...")
        if not hasattr(model, 'predict_proba'):
            raise TypeError(
                f"Loaded model type {type(model)} does not support predict_proba.")
        prediction_proba_all = model.predict_proba(data_processed)
        if prediction_proba_all.shape[1] < 2: raise ValueError(
            f"Model predict_proba returned unexpected shape: {prediction_proba_all.shape}")
        prediction_proba = prediction_proba_all[:, 1]
        threshold = 0.5
        prediction = (prediction_proba >= threshold).astype(int)[0]
        prediction_proba_val = float(prediction_proba[0])
        print(f"Prediction successful. Probability: {prediction_proba_val:.4f}, Class: {prediction}")
    except Exception as e:
        print(f"ERROR during prediction: {e}");
        traceback.print_exc()
        raise RuntimeError(f"Error during prediction: {e}")

    return prediction, prediction_proba_val


# Пример использования
if __name__ == '__main__':
    print("\n--- Running prediction module test ---")
    test_input_final = {
        'visit_number': 1, 'visit_hour': 10, 'visit_dayofweek': 2, 'visit_month': 3,
        'is_weekend': 0, 'is_night_hours': 0, 'n_hits': 15, 'n_unique_pages': 5,
        'session_duration_sec': 180.0, 'avg_time_per_hit_sec': 12.0,
        'is_single_page_session': 0, 'is_short_session_5s': 0,
        'utm_source_limited': 'ZpYIoDJMcFzVoPFsHGJL', 'utm_medium': 'banner',
        'utm_campaign_limited': 'LEoPHuyFvzoNfnzGgfcd', 'utm_adcontent_limited': 'vCIpmpaGBnIQhyYNkXqp',
        'device_category': 'mobile', 'device_os': 'Android', 'device_brand_limited': 'Samsung',
        'device_browser': 'Chrome', 'geo_city_limited': 'Krasnodar', 'geo_country': 'Russia',
        'traffic_type': 'paid', 'device_os_browser_limited': 'Android_Chrome',
    }
    print(f"\nTest input sample keys: {list(test_input_final.keys())}")
    try:
        load_prediction_assets()
        if required_columns:
            print(f"Preprocessor expects columns: {required_columns}")
            input_keys_set = set(test_input_final.keys())
            required_keys_set = set(required_columns)
            if input_keys_set != required_keys_set:
                print("\nWARNING: Keys in test_input_final do not exactly match preprocessor expectations!")
                if required_keys_set - input_keys_set: print(f"Missing: {required_keys_set - input_keys_set}")
                if input_keys_set - required_keys_set: print(f"Extra: {input_keys_set - required_keys_set}")
            else:
                print("Test input keys match preprocessor expectations.")
        pred_class, pred_proba = predict_single(test_input_final)
        print(f"\n--- Test Prediction Result ---")
        print(f"Predicted Class: {pred_class}")
        print(f"Predicted Probability (Target=1): {pred_proba:.4f}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Ensure model files ('preprocessor_final.joblib', 'model_final.joblib') exist in '../models/'.")
    except Exception as e:
        print(f"\nAn error occurred during the prediction test: {e}")
        traceback.print_exc()
