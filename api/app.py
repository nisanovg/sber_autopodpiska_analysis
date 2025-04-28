from flask import Flask, request, jsonify
import sys
import os
import traceback
import time

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Импортируем модуль предсказания
try:
    import prediction  # Используем обновленный prediction.py
except ImportError as e:
    print(f"ERROR: Could not import prediction module: {e}.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during prediction module import: {e}")
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)

# --- Загрузка моделей при старте ---
try:
    print("API Initializing: Loading prediction assets...")
    # Используем финальные имена файлов
    prediction.load_prediction_assets(
        preprocessor_filename='preprocessor_final.joblib',
        model_filename='model_final.joblib'
    )
    if prediction.model is None or prediction.preprocessor is None:
        raise RuntimeError("Model or preprocessor failed to load.")
    print("API Ready: Prediction assets loaded.")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR during API init: {e}")
    print("Ensure final model files exist in the 'models' directory.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR loading models during API init: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Эндпоинты ---

@app.route('/')
def home():
    """Проверка статуса API."""
    status = "OK" if prediction.model and prediction.preprocessor else "Error (Models not loaded)"
    model_type = type(prediction.model).__name__ if prediction.model else "N/A"
    return jsonify({
        "message": "Sber Autopodpiska Prediction API",
        "status": status,
        "loaded_model_type": model_type
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Эндпоинт для предсказаний."""
    endpoint_start_time = time.time()
    print("\nReceived request on /predict")

    if not request.is_json:
        print("Error: Request content type is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        input_data = request.get_json()
        if not isinstance(input_data, dict):
            print(f"Error: Invalid JSON payload type: {type(input_data)}. Expected object.")
            return jsonify({"error": "JSON payload must be an object/dictionary"}), 400
    except Exception as e:
        print(f"Error getting JSON data from request: {e}")
        return jsonify({"error": f"Failed to parse JSON data: {e}"}), 400

    # Валидация входных данных
    if prediction.required_columns:
        input_keys = set(input_data.keys())
        required_keys = set(prediction.required_columns)
        if not required_keys.issubset(input_keys):
            missing = required_keys - input_keys
            print(f"Error: Missing required fields in input JSON: {missing}")
            return jsonify({"error": f"Missing required fields: {list(missing)}"}), 400

    try:
        pred_class, pred_proba = prediction.predict_single(input_data)
    except FileNotFoundError as e:
        print(f"Internal Server Error: Prediction assets not found: {e}")
        return jsonify({"error": "Prediction assets not found."}), 500
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error during prediction processing: {e}")
        error_message = f"Prediction processing failed: {e}"
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return jsonify({"error": error_message}), status_code
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred."}), 500

    endpoint_end_time = time.time()
    response_time = endpoint_end_time - endpoint_start_time

    response = {
        'predicted_class': int(pred_class),
        'predicted_probability_target_1': float(pred_proba),
        'processing_time_seconds': round(response_time, 4)
    }
    print(f"Prediction response: {response}")
    return jsonify(response), 200


# --- Запуск приложения ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
