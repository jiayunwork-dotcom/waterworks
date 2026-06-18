import pandas as pd
import numpy as np
from scipy import signal


def calculate_correlation(df):
    if df is None:
        return pd.DataFrame()
    
    numeric_cols = []
    for col in df.columns:
        if col == '时间':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            try:
                pd.to_numeric(df[col])
                numeric_cols.append(col)
            except Exception:
                pass
    
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    corr_matrix = numeric_df.corr(method='pearson')
    
    return corr_matrix


def get_correlation_heatmap_data(corr_matrix):
    if corr_matrix.empty:
        return pd.DataFrame()
    
    heatmap_data = []
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            value = corr_matrix.loc[i, j]
            if pd.isna(value):
                continue
            
            if abs(value) > 0.7:
                color = 'red'
            elif abs(value) > 0.4:
                color = 'yellow'
            else:
                color = 'green'
            
            heatmap_data.append({
                'x': i,
                'y': j,
                'value': round(value, 4),
                'color': color,
            })
    
    return pd.DataFrame(heatmap_data)


def cross_correlation(x, y, max_lag=24):
    x = pd.to_numeric(x, errors='coerce').ffill().values
    y = pd.to_numeric(y, errors='coerce').ffill().values
    
    x = (x - np.mean(x)) / (np.std(x) * len(x))
    y = (y - np.mean(y)) / np.std(y)
    
    correlation = signal.correlate(x, y, mode='full')
    lags = signal.correlation_lags(len(x), len(y), mode='full')
    
    mid = len(lags) // 2
    start = max(0, mid - max_lag)
    end = min(len(lags), mid + max_lag + 1)
    
    return lags[start:end], correlation[start:end]


def find_best_lag(x, y, max_lag=24):
    lags, corr = cross_correlation(x, y, max_lag)
    
    positive_lags = lags >= 0
    best_idx = np.argmax(corr[positive_lags])
    best_lag = lags[positive_lags][best_idx]
    best_corr = corr[positive_lags][best_idx]
    
    return int(best_lag), round(float(best_corr), 4)


def detect_turbidity_shocks(df, threshold=0.5):
    if df is None or '源水浊度' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['源水浊度'] = pd.to_numeric(df['源水浊度'], errors='coerce')
    df = df.dropna(subset=['源水浊度'])
    
    shocks = []
    
    for i in range(1, len(df)):
        prev_val = df['源水浊度'].iloc[i-1]
        curr_val = df['源水浊度'].iloc[i]
        
        if prev_val == 0:
            continue
        
        change_rate = abs(curr_val - prev_val) / prev_val
        
        if change_rate > threshold:
            shocks.append({
                '时间': df['时间'].iloc[i],
                '前一时刻浊度(NTU)': round(prev_val, 3),
                '当前浊度(NTU)': round(curr_val, 3),
                '变化率(%)': round(change_rate * 100, 2),
                '变化方向': '上升' if curr_val > prev_val else '下降',
            })
    
    if shocks:
        return pd.DataFrame(shocks)
    return pd.DataFrame()


def analyze_shock_response(df, shocks_df, dosage_col='混凝剂投加量', response_window=6):
    if shocks_df.empty or dosage_col not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df[dosage_col] = pd.to_numeric(df[dosage_col], errors='coerce')
    
    response_analysis = []
    
    for _, shock in shocks_df.iterrows():
        shock_time = shock['时间']
        
        shock_idx = df[df['时间'] == shock_time].index
        if len(shock_idx) == 0:
            continue
        
        shock_idx = shock_idx[0]
        
        before_window_start = max(0, shock_idx - response_window)
        before_dosage = df.iloc[before_window_start:shock_idx][dosage_col].mean()
        
        after_window_end = min(len(df), shock_idx + response_window + 1)
        after_dosage = df.iloc[shock_idx+1:after_window_end][dosage_col].mean()
        
        if pd.isna(before_dosage) or pd.isna(after_dosage):
            continue
        
        adjustment = after_dosage - before_dosage
        adjustment_pct = (adjustment / before_dosage * 100) if before_dosage != 0 else 0
        
        response_analysis.append({
            '时间': shock_time,
            '浊度变化率(%)': shock['变化率(%)'],
            '变化方向': shock['变化方向'],
            '震前平均投药量(mg/L)': round(before_dosage, 3),
            '震后平均投药量(mg/L)': round(after_dosage, 3),
            '投药量调整量(mg/L)': round(adjustment, 3),
            '调整幅度(%)': round(adjustment_pct, 2),
            '响应是否充分': '是' if (shock['变化方向'] == '上升' and adjustment > 0) or 
                               (shock['变化方向'] == '下降' and adjustment < 0) else '否',
        })
    
    if response_analysis:
        return pd.DataFrame(response_analysis)
    return pd.DataFrame()
