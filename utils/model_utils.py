import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import os
from datetime import datetime


FEATURE_COLS_CN = ['源水浊度', '源水pH', '源水温度', '源水氨氮', '源水有机物']
TARGET_COL_CN = '混凝剂投加量'

MODEL_TYPES = {
    '多元线性回归': 'linear',
    '随机森林': 'random_forest',
    'XGBoost': 'xgboost',
}


def prepare_training_data(df):
    if df is None:
        return None, None
    
    feature_cols = [c for c in FEATURE_COLS_CN if c in df.columns]
    target_col = TARGET_COL_CN if TARGET_COL_CN in df.columns else None
    
    if not target_col or len(feature_cols) < 3:
        return None, None
    
    data = df[feature_cols + [target_col]].copy()
    for col in feature_cols + [target_col]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    
    if len(data) < 20:
        return None, None
    
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y


def train_model(X, y, model_type='random_forest', test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
        )
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            objective='reg:squarederror',
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = {
        'R²': round(r2_score(y_train, y_train_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
        'MAE': round(mean_absolute_error(y_train, y_train_pred), 4),
    }
    
    test_metrics = {
        'R²': round(r2_score(y_test, y_test_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
        'MAE': round(mean_absolute_error(y_test, y_test_pred), 4),
    }
    
    residuals = y_test - y_test_pred
    
    return {
        'model': model,
        'model_type': model_type,
        'feature_cols': list(X.columns),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'residuals': residuals,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
    }


def predict_dosage(model_result, features, confidence=0.95):
    model = model_result['model']
    feature_cols = model_result['feature_cols']
    residuals = model_result['residuals']
    
    input_df = pd.DataFrame([features])[feature_cols]
    prediction = model.predict(input_df)[0]
    
    residual_std = np.std(residuals)
    z_score = 1.96 if confidence == 0.95 else 1.645
    margin = z_score * residual_std
    
    return {
        'prediction': round(prediction, 2),
        'lower': round(max(0, prediction - margin), 2),
        'upper': round(prediction + margin, 2),
        'confidence': confidence,
    }


def compare_models(X, y):
    results = {}
    
    for name, model_type in MODEL_TYPES.items():
        try:
            result = train_model(X, y, model_type=model_type)
            results[name] = result
        except Exception as e:
            results[name] = None
    
    comparison = []
    for name, result in results.items():
        if result is None:
            continue
        comparison.append({
            '模型': name,
            '训练集R²': result['train_metrics']['R²'],
            '测试集R²': result['test_metrics']['R²'],
            '训练集RMSE': result['train_metrics']['RMSE'],
            '测试集RMSE': result['test_metrics']['RMSE'],
            '训练集MAE': result['train_metrics']['MAE'],
            '测试集MAE': result['test_metrics']['MAE'],
        })
    
    return pd.DataFrame(comparison), results


def save_model(model_result, version_dir='models'):
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_type = model_result['model_type']
    filename = f"{model_type}_{timestamp}.pkl"
    filepath = os.path.join(version_dir, filename)
    
    save_data = {
        'model': model_result['model'],
        'model_type': model_result['model_type'],
        'feature_cols': model_result['feature_cols'],
        'train_metrics': model_result['train_metrics'],
        'test_metrics': model_result['test_metrics'],
        'residuals': model_result['residuals'].tolist() if hasattr(model_result['residuals'], 'tolist') else model_result['residuals'],
        'y_test': model_result['y_test'].tolist() if hasattr(model_result['y_test'], 'tolist') else model_result['y_test'],
        'y_test_pred': model_result['y_test_pred'].tolist() if hasattr(model_result['y_test_pred'], 'tolist') else model_result['y_test_pred'],
        'timestamp': timestamp,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    return filepath, timestamp


def load_model_versions(version_dir='models'):
    if not os.path.exists(version_dir):
        return []
    
    versions = []
    files = sorted([f for f in os.listdir(version_dir) if f.endswith('.pkl')], reverse=True)
    
    for filename in files[:3]:
        filepath = os.path.join(version_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            versions.append({
                'filename': filename,
                'filepath': filepath,
                'model_type': data.get('model_type', 'unknown'),
                'timestamp': data.get('timestamp', ''),
                'train_metrics': data.get('train_metrics', {}),
                'test_metrics': data.get('test_metrics', {}),
                'y_test': data.get('y_test', []),
                'y_test_pred': data.get('y_test_pred', []),
                'data': data,
            })
        except Exception:
            continue
    
    return versions


def get_feature_importance(model_result):
    model = model_result['model']
    model_type = model_result['model_type']
    feature_cols = model_result['feature_cols']
    
    if model_type in ['random_forest', 'xgboost']:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances,
        })
        importance_df = importance_df.sort_values('importance', ascending=True).reset_index(drop=True)
        return {
            'type': 'importance',
            'data': importance_df,
        }
    elif model_type == 'linear':
        coefs = model.coef_
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs),
        })
        coef_df = coef_df.sort_values('abs_coefficient', ascending=True).reset_index(drop=True)
        return {
            'type': 'coefficient',
            'data': coef_df,
        }
    else:
        return None


def find_similar_conditions(df, features, feature_cols, top_k=5):
    if df is None or len(df) == 0:
        return None
    
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) != len(feature_cols):
        return None
    
    data = df[available_cols + [TARGET_COL_CN]].copy()
    for col in available_cols + [TARGET_COL_CN]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    if len(data) == 0:
        return None
    
    scaler = StandardScaler()
    historical_scaled = scaler.fit_transform(data[available_cols])
    
    input_df = pd.DataFrame([[features[c] for c in available_cols]], columns=available_cols)
    input_scaled = scaler.transform(input_df)
    
    distances = np.sqrt(np.sum((historical_scaled - input_scaled) ** 2, axis=1))
    
    data['distance'] = distances
    similar_records = data.sort_values('distance', ascending=True).head(top_k).copy()
    similar_records['distance'] = similar_records['distance'].round(2)
    
    return similar_records[available_cols + [TARGET_COL_CN, 'distance']]
