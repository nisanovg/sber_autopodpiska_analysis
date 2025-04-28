import pandas as pd
import numpy as np
import time
import joblib
import os
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, RocCurveDisplay,
    precision_recall_curve, auc, classification_report,
    precision_score, recall_score, f1_score
)
import warnings
import traceback

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- Вспомогательные функции ---

def get_feature_names(column_transformer):
    """Возвращает имена признаков после ColumnTransformer (улучшенная версия)."""
    output_features = []
    try:
        if hasattr(column_transformer, 'get_feature_names_out'):
            output_features = column_transformer.get_feature_names_out()
            return list(output_features)  # Возвращаем как список
    except Exception as e:
        pass  # Продолжаем с ручным парсингом

    # Ручной парсинг (фоллбэк)
    output_features = []
    for name, pipe_or_trans, features in column_transformer.transformers_:
        if pipe_or_trans == 'drop' or len(features) == 0: continue
        if name == 'remainder':
            if column_transformer.remainder == 'passthrough':
                try:
                    # Используем feature_names_in_, если он есть у column_transformer
                    if hasattr(column_transformer, 'feature_names_in_'):
                        remainder_features = np.array(column_transformer.feature_names_in_)[features].tolist()
                    else:  # Иначе берем исходные имена из features (если это строки)
                        if all(isinstance(f, str) for f in features):
                            remainder_features = features
                        else:  # Фоллбэк на индексы
                            remainder_features = [f"remainder__{i}" for i in range(len(features))]
                    output_features.extend(remainder_features)
                except (AttributeError, IndexError, KeyError, TypeError):
                    output_features.extend([f"remainder__{i}" for i in range(len(features))])
            continue

        # Получаем имена от последнего шага пайплайна или от трансформера
        current_transformer = pipe_or_trans.steps[-1][1] if isinstance(pipe_or_trans, Pipeline) else pipe_or_trans
        if hasattr(current_transformer, 'get_feature_names_out'):
            try:
                # Передаем исходные имена фичей этого шага
                step_feature_names = current_transformer.get_feature_names_out(features)
                output_features.extend(step_feature_names)
            except Exception:  # Если get_feature_names_out не сработал
                output_features.extend(features)  # Фоллбэк на исходные
        else:  # Если нет get_feature_names_out
            output_features.extend(features)  # Используем исходные имена

    return output_features


def create_preprocessor(numeric_features, categorical_features):
    """Создает пайплайн препроцессинга с OHE + Scaler (n_jobs=1)."""
    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
        return ColumnTransformer(transformers=[], remainder='passthrough')

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',
        n_jobs=1  # Последовательная обработка для экономии памяти
    )
    return preprocessor


def plot_roc_pr_curves(y_true, y_pred_proba, model_name, ax_roc, ax_pr):
    """Рисует ROC и PR кривые на предоставленных осях."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name).plot(ax=ax_roc)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, label=f'{model_name} (PR AUC = {pr_auc:.3f})')
        return roc_auc, pr_auc
    except Exception as e:
        print(f"Error plotting curves for {model_name}: {e}")
        return np.nan, np.nan


def get_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """Рассчитывает основные метрики классификации."""
    if y_pred_proba is None or len(np.unique(y_true)) < 2:
        return {key: 0.0 for key in ['ROC-AUC', 'PR-AUC', 'Precision (0)', 'Recall (0)', 'F1-score (0)',
                                     'Precision (1)', 'Recall (1)', 'F1-score (1)', 'Accuracy']}
    try:
        y_pred_class = (y_pred_proba >= threshold).astype(int)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        report = classification_report(y_true, y_pred_class, output_dict=True, zero_division=0)
        metrics = {
            'ROC-AUC': roc_auc, 'PR-AUC': pr_auc,
            'Precision (0)': report.get('0', {}).get('precision', 0.0),
            'Recall (0)': report.get('0', {}).get('recall', 0.0),
            'F1-score (0)': report.get('0', {}).get('f1-score', 0.0),
            'Precision (1)': report.get('1', {}).get('precision', 0.0),
            'Recall (1)': report.get('1', {}).get('recall', 0.0),
            'F1-score (1)': report.get('1', {}).get('f1-score', 0.0),
            'Accuracy': report.get('accuracy', 0.0)
        }
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {key: 0.0 for key in ['ROC-AUC', 'PR-AUC', 'Precision (0)', 'Recall (0)', 'F1-score (0)',
                                     'Precision (1)', 'Recall (1)', 'F1-score (1)', 'Accuracy']}


# --- Функции для обучения и оценки ---

def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name,
                         preprocessor=None, fit_params=None):
    """
    Обучает одну модель (или пайплайн) и возвращает метрики и обученный объект.
    """
    print(f"\n--- Training and Evaluating: {model_name} ---")
    start_time = time.time()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)]) if preprocessor and preprocessor.transformers else model

    # Обучение
    try:
        if fit_params:
            if isinstance(pipeline, Pipeline):
                fit_params_adj = {f'classifier__{k}': v for k, v in fit_params.items()}
                if 'classifier__eval_set' in fit_params_adj:
                    eval_data = fit_params_adj['classifier__eval_set'][0]
                    if len(eval_data) == 2:
                        X_eval, y_eval = eval_data
                        if 'preprocessor' in pipeline.named_steps:
                            X_eval_transformed = pipeline.named_steps['preprocessor'].transform(X_eval)
                            fit_params_adj['classifier__eval_set'] = [(X_eval_transformed, y_eval)]
                        else:
                            fit_params_adj['classifier__eval_set'] = [(X_eval, y_eval)]
                    else:
                        del fit_params_adj['classifier__eval_set']
                pipeline.fit(X_train, y_train, **fit_params_adj)
            else:
                pipeline.fit(X_train, y_train, **fit_params)
        else:
            pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"ERROR during training {model_name}: {e}");
        traceback.print_exc()
        return None, None, None
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    # Предсказание
    y_pred_proba = None
    try:
        if not X_test.empty: y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"ERROR during prediction for {model_name}: {e}");
        traceback.print_exc()
        return pipeline, get_classification_metrics(y_test, None), None

    # Метрики
    metrics = get_classification_metrics(y_test, y_pred_proba)
    print(f"Test ROC-AUC: {metrics.get('ROC-AUC', 0.0):.4f}, Test PR-AUC: {metrics.get('PR-AUC', 0.0):.4f}")
    print(f"Test F1 (0): {metrics.get('F1-score (0)', 0.0):.4f}, Test F1 (1): {metrics.get('F1-score (1)', 0.0):.4f}")

    # Важность признаков
    feature_importance_df = None
    try:
        model_in_pipeline = pipeline.named_steps['classifier'] if isinstance(pipeline, Pipeline) else pipeline
        feature_names_out = []
        if isinstance(pipeline, Pipeline) and 'preprocessor' in pipeline.named_steps:
            preprocessor_in_pipeline = pipeline.named_steps['preprocessor']
            if preprocessor_in_pipeline.transformers_:  # Проверяем, что трансформеры не пустые
                feature_names_out = get_feature_names(preprocessor_in_pipeline)  # Используем нашу улучшенную функцию
            else:
                feature_names_out = X_train.columns.tolist()  # Если препроцессор пуст
        else:
            feature_names_out = X_train.columns.tolist()  # Если модель без пайплайна

        importances = None
        if hasattr(model_in_pipeline, 'feature_importances_'):
            importances = model_in_pipeline.feature_importances_
        elif hasattr(model_in_pipeline, 'coef_'):
            importances = model_in_pipeline.coef_[0]

        if isinstance(importances, np.ndarray) and feature_names_out and len(importances) == len(feature_names_out):
            feature_importance_df = pd.DataFrame({
                'feature': feature_names_out,
                'importance': importances
            }).sort_values(by='importance', key=abs, ascending=False).reset_index(drop=True)
        elif isinstance(importances, np.ndarray):  # Если importances есть, но длины не совпали
            print(
                f"Warning: Mismatch in feature importance length ({len(importances)}) and feature names length ({len(feature_names_out)}). Cannot create importance DataFrame.")

    except Exception as e:
        print(f"Could not extract feature importances for {model_name}: {e}")

    return pipeline, metrics, feature_importance_df


# --- Целевая функция Optuna ---

def objective_lgbm(trial, X_train, y_train, numeric_features, categorical_features, n_folds=3, random_state=42):
    """Целевая функция для подбора гиперпараметров LGBM с помощью Optuna (возвращает ROC-AUC)."""
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    if not preprocessor.transformers: return 0.0

    # --- ИЗМЕНЕНИЕ: Скорректирован диапазон scale_pos_weight ---
    # Соотношение классов ~97/3, т.е. вес для класса 1 должен быть ~32
    # Задаем диапазон вокруг этого значения
    scale_pos_weight_val = trial.suggest_float('scale_pos_weight', 15.0, 50.0)  # Примерный диапазон 15-50
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt',
        'random_state': random_state, 'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'scale_pos_weight': scale_pos_weight_val  # Используем подобранное значение
    }
    cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        try:
            preprocessor_fold = create_preprocessor(numeric_features, categorical_features)
            X_train_fold_processed = preprocessor_fold.fit_transform(X_train_fold)
            X_val_fold_processed = preprocessor_fold.transform(X_val_fold)
            model = LGBMClassifier(**params)
            model.fit(X_train_fold_processed, y_train_fold,
                      eval_set=[(X_val_fold_processed, y_val_fold)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            y_pred_proba_val = model.predict_proba(X_val_fold_processed)[:, 1]
            roc_auc_val = roc_auc_score(y_val_fold, y_pred_proba_val)
            cv_scores.append(roc_auc_val)
            trial.report(roc_auc_val, fold)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        except Exception as e:
            print(f"Error in Optuna trial {trial.number} fold {fold + 1}: {e}")
            return 0.0
    mean_roc_auc = np.mean(cv_scores) if cv_scores else 0.0
    # Выводим метрику для триала
    print(f"Optuna Trial {trial.number} finished. Mean ROC-AUC: {mean_roc_auc:.5f}")
    return mean_roc_auc


# --- Основная функция обучения ---

def run_training_pipeline(X, y, model_config, preprocessor_config):
    """
    Запускает полный пайплайн обучения и сравнения моделей, включая Optuna для LGBM.
    (Версия с исправленной важностью фичей и новым диапазоном scale_pos_weight)
    """
    print("--- Starting Training Pipeline ---")
    start_pipeline_time = time.time()
    random_state = model_config.get('random_state', 42)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_config.get('test_size', 0.25),
        random_state=random_state,
        stratify=y
    )

    # --- Препроцессор OHE (с n_jobs=1) ---
    numeric_features = preprocessor_config.get('numeric_features', [])
    categorical_features = preprocessor_config.get('categorical_features', [])
    if not numeric_features and not categorical_features:
        print("CRITICAL WARNING: No features specified for training!")

    preprocessor_ohe = create_preprocessor(numeric_features, categorical_features)
    print("Fitting OHE preprocessor...")
    fit_start_time = time.time()
    if preprocessor_ohe.transformers:
        preprocessor_ohe.fit(X_train)
        fit_duration = time.time() - fit_start_time
        print(f"OHE preprocessor fitted in {fit_duration:.2f} seconds.")

    # --- Подготовка данных для CatBoost ---
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    cat_features_cb = categorical_features
    cat_features_indices = []
    if cat_features_cb:
        valid_cat_features_cb = [col for col in cat_features_cb if col in X_train_cb.columns]
        for col in valid_cat_features_cb:
            X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing').astype('category')
            X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing').astype('category')
        cat_features_indices = [X_train_cb.columns.get_loc(col) for col in valid_cat_features_cb]
    num_features_cb = numeric_features
    if num_features_cb:
        valid_num_features_cb = [col for col in num_features_cb if col in X_train_cb.columns]
        if valid_num_features_cb:
            imputer_num_cb = SimpleImputer(strategy='median')
            X_train_cb[valid_num_features_cb] = imputer_num_cb.fit_transform(X_train_cb[valid_num_features_cb])
            X_test_cb[valid_num_features_cb] = imputer_num_cb.transform(X_test_cb[valid_num_features_cb])

    # --- Optuna для LGBM ---
    best_params_lgbm = {}
    if preprocessor_ohe.transformers:
        print("\nRunning Optuna for LightGBM...")
        start_optuna_time = time.time()
        study_lgbm = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        objective_lgbm_lambda = lambda trial: objective_lgbm(
            trial, X_train, y_train, numeric_features, categorical_features,
            n_folds=model_config.get('optuna_cv_folds', 3),
            random_state=random_state
        )
        optuna_trials = model_config.get('optuna_n_trials', 50)  # Увеличили число попыток по умолчанию
        try:
            study_lgbm.optimize(objective_lgbm_lambda, n_trials=optuna_trials, n_jobs=1)  # n_jobs=1
            best_params_lgbm = study_lgbm.best_params
            optuna_duration = time.time() - start_optuna_time
            print(f"Optuna finished {optuna_trials} trials in {optuna_duration:.2f} seconds.")
            print(f"Best LGBM params found (ROC-AUC={study_lgbm.best_value:.5f}): {best_params_lgbm}")
        except Exception as optuna_e:
            print(f"Optuna optimization failed: {optuna_e}. Using default LGBM params.")
            traceback.print_exc()
            best_params_lgbm = {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'scale_pos_weight': 1.0}
    else:
        print("Skipping Optuna as OHE preprocessor is empty.")

    # --- Определение моделей ---
    models_to_train = {}
    neg_train, pos_train = np.bincount(y_train)
    scale_pos_weight_train = neg_train / pos_train if pos_train > 0 and neg_train > 0 else 1
    print(f"\nCalculated scale_pos_weight for CatBoost: {scale_pos_weight_train:.2f}")

    models_to_train['Dummy'] = DummyClassifier(strategy='stratified', random_state=random_state)
    if preprocessor_ohe.transformers:
        models_to_train['LogisticRegression'] = LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                   random_state=random_state, max_iter=1000)
        if not best_params_lgbm:
            best_params_lgbm = {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
                                'scale_pos_weight': scale_pos_weight_train}
        models_to_train['LightGBM_Optuna'] = LGBMClassifier(objective='binary', metric='auc', random_state=random_state,
                                                            n_jobs=-1, **best_params_lgbm)
    models_to_train['CatBoost'] = CatBoostClassifier(
        random_state=random_state, cat_features=cat_features_indices,
        scale_pos_weight=scale_pos_weight_train, eval_metric='AUC',
        iterations=model_config.get('catboost_iterations', 600),  # Увеличили итерации
        learning_rate=model_config.get('catboost_learning_rate', 0.04),
        early_stopping_rounds=model_config.get('catboost_early_stopping', 50),
        verbose=0
    )

    # --- Обучение и оценка ---
    results = {}
    feature_importances = {}
    trained_models = {}
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

    for name, model in models_to_train.items():
        fit_params = None
        current_preprocessor = None
        current_X_train = X_train
        current_X_test = X_test
        if name in ['LogisticRegression', 'LightGBM_Optuna']:
            current_preprocessor = preprocessor_ohe
        elif name == 'CatBoost':
            current_X_train = X_train_cb
            current_X_test = X_test_cb
            if not current_X_test.empty:
                fit_params = {'eval_set': [(current_X_test, y_test)], 'verbose': 0}
            else:
                fit_params = {'verbose': 0}
        if name in ['LogisticRegression', 'LightGBM_Optuna'] and not preprocessor_ohe.transformers: continue

        eval_results = train_evaluate_model(model, current_X_train, y_train, current_X_test, y_test, name,
                                            preprocessor=current_preprocessor, fit_params=fit_params)

        if eval_results and eval_results[0] is not None:
            pipeline, metrics, f_imp_df = eval_results
            results[name] = metrics
            feature_importances[name] = f_imp_df
            trained_models[name] = pipeline
            try:
                if not current_X_test.empty:
                    y_pred_proba_test = pipeline.predict_proba(current_X_test)[:, 1]
                    plot_roc_pr_curves(y_test, y_pred_proba_test, name, ax_roc, ax_pr)
            except Exception as plot_e:
                print(f"Could not plot curves for {name}: {plot_e}")

    # --- Финализация графиков ---
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    ax_roc.grid(True)
    plt.show()
    ax_pr.set_title('Precision-Recall Curves Comparison')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()
    ax_pr.grid(True)
    plt.show()

    # --- Вывод и сохранение ---
    if not results:
        print("\nCRITICAL: No models were successfully trained and evaluated.")
        pipeline_duration = time.time() - start_pipeline_time
        print(f"\n--- Training Pipeline Finished (No Results) in {pipeline_duration:.2f} seconds ---")
        return pd.DataFrame(), {}, {}

    results_df = pd.DataFrame(results).T.sort_values(by='ROC-AUC', ascending=False)
    print("\n--- Model Comparison (Test Set Metrics) ---")
    cols_to_show = ['ROC-AUC', 'PR-AUC', 'F1-score (0)', 'F1-score (1)', 'Accuracy']
    print(results_df[[col for col in cols_to_show if col in results_df.columns]].round(4))
    results_no_dummy = results_df.drop('Dummy', errors='ignore')
    if results_no_dummy.empty:
        best_model_name = 'Dummy'
    else:
        best_model_name = results_no_dummy.index[0]  # Лучшая по ROC-AUC
    print(f"\nBest model (excluding Dummy) based on ROC-AUC: {best_model_name}")

    # --- Сохранение артефактов ---
    model_to_save_name = None
    preprocessor_to_save = None
    final_model_to_save = None
    ohe_compatible_models = {k: v for k, v in results.items() if k in ['LightGBM_Optuna', 'LogisticRegression']}
    if ohe_compatible_models:
        best_ohe_model_name = pd.DataFrame(ohe_compatible_models).T.sort_values(by='ROC-AUC', ascending=False).index[0]
        model_to_save_name = best_ohe_model_name
        if best_model_name == 'CatBoost' and results[best_model_name]['ROC-AUC'] > results[best_ohe_model_name][
            'ROC-AUC'] + 0.01:
            print(
                f"Note: CatBoost is significantly better (ROC-AUC={results['CatBoost']['ROC-AUC']:.4f}), but saving OHE-compatible: {model_to_save_name} (ROC-AUC={results[model_to_save_name]['ROC-AUC']:.4f})")
        else:
            print(f"Selected OHE-compatible model for saving: {model_to_save_name}")
    elif best_model_name == 'CatBoost':
        model_to_save_name = 'CatBoost'
        print("Warning: Only CatBoost available or significantly better. Saving CatBoost.")
    elif best_model_name == 'Dummy':
        print("Only Dummy model available. Nothing to save.")
    else:
        print("No suitable model found to save.")

    if model_to_save_name and model_to_save_name in trained_models:
        saved_pipeline = trained_models[model_to_save_name]
        models_dir = model_config.get('models_dir', '../models')
        preprocessor_filename = model_config.get('preprocessor_filename',
                                                 'preprocessor_final.joblib')  # Используем финальные имена
        model_filename = model_config.get('model_filename', 'model_final.joblib')  # Используем финальные имена
        preprocessor_path = os.path.join(models_dir, preprocessor_filename)
        model_path = os.path.join(models_dir, model_filename)
        os.makedirs(models_dir, exist_ok=True)
        if isinstance(saved_pipeline, Pipeline):
            preprocessor_to_save = saved_pipeline.named_steps.get('preprocessor')
            final_model_to_save = saved_pipeline.named_steps.get('classifier')
            if preprocessor_to_save:
                print(f"Saving OHE preprocessor to {preprocessor_path}")
                joblib.dump(preprocessor_to_save, preprocessor_path)
            if final_model_to_save:
                print(f"Saving trained {model_to_save_name} model to {model_path}")
                joblib.dump(final_model_to_save, model_path)
        elif model_to_save_name == 'CatBoost':
            final_model_to_save = saved_pipeline
            print(f"Saving trained {model_to_save_name} model to {model_path}")
            joblib.dump(final_model_to_save, model_path)
            print("Note: Preprocessor for CatBoost was not saved.")
        else:  # Dummy
            print(f"Saving trained {model_to_save_name} model to {model_path}")
            joblib.dump(saved_pipeline, model_path)
    else:
        print("Could not save model artifacts.")

    pipeline_duration = time.time() - start_pipeline_time
    print(f"\n--- Training Pipeline Finished in {pipeline_duration:.2f} seconds ---")
    return results_df, feature_importances, trained_models
