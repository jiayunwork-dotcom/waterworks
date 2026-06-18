import pandas as pd
import numpy as np
from io import StringIO
import streamlit as st


COLUMN_MAPPING = {
    '时间': 'timestamp',
    '源水浊度': 'raw_turbidity',
    '源水pH': 'raw_ph',
    '源水温度': 'raw_temperature',
    '源水氨氮': 'raw_ammonia',
    '源水COD': 'raw_cod',
    '源水有机物': 'raw_uv254',
    '混凝剂投加量': 'coagulant_dosage',
    '沉后浊度': 'settled_turbidity',
    '滤后浊度': 'filtered_turbidity',
    '出水浊度': 'effluent_turbidity',
    '出水余氯': 'effluent_chlorine',
    '出水pH': 'effluent_ph',
}

REVERSE_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

STANDARD_LIMITS = {
    'effluent_turbidity': {'min': 0, 'max': 1.0, 'unit': 'NTU', 'name': '出水浊度'},
    'effluent_chlorine': {'min': 0.3, 'max': 4.0, 'unit': 'mg/L', 'name': '出水余氯'},
    'effluent_ph': {'min': 6.5, 'max': 8.5, 'unit': '', 'name': '出水pH'},
}

PROCESS_STAGES = [
    {'name': '取水', 'metrics': ['raw_turbidity', 'raw_ph', 'raw_temperature', 'raw_ammonia', 'raw_cod', 'raw_uv254']},
    {'name': '混凝', 'metrics': ['coagulant_dosage']},
    {'name': '沉淀', 'metrics': ['settled_turbidity']},
    {'name': '过滤', 'metrics': ['filtered_turbidity']},
    {'name': '消毒', 'metrics': ['effluent_chlorine']},
    {'name': '清水池', 'metrics': ['effluent_turbidity', 'effluent_ph']},
    {'name': '供水', 'metrics': []},
]


def load_data(file):
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None


def process_data(df):
    if df is None:
        return None
    
    df = df.copy()
    
    if '时间' in df.columns:
        df['时间'] = pd.to_datetime(df['时间'])
        df = df.sort_values('时间')
        df = df.reset_index(drop=True)
    
    return df


def get_statistics(df):
    if df is None:
        return None
    
    stats = []
    for col in df.columns:
        if col == '时间':
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        missing_rate = series.isna().mean() * 100
        
        stats.append({
            '指标': col,
            '均值': round(series.mean(), 3) if not pd.isna(series.mean()) else 'N/A',
            '标准差': round(series.std(), 3) if not pd.isna(series.std()) else 'N/A',
            '最小值': round(series.min(), 3) if not pd.isna(series.min()) else 'N/A',
            '最大值': round(series.max(), 3) if not pd.isna(series.max()) else 'N/A',
            '缺失率(%)': round(missing_rate, 2),
            '状态': '正常' if missing_rate < 10 else ('警告' if missing_rate < 30 else '严重'),
        })
    
    return pd.DataFrame(stats)


def get_missing_color(missing_rate):
    if missing_rate < 10:
        return 'green'
    elif missing_rate < 30:
        return 'orange'
    else:
        return 'red'


def get_latest_values(df):
    if df is None or len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    result = {}
    for col in df.columns:
        if col == '时间':
            result['timestamp'] = latest[col]
        else:
            result[col] = latest[col]
    return result


def check_standard(value, metric):
    if metric not in STANDARD_LIMITS:
        return True
    limits = STANDARD_LIMITS[metric]
    if pd.isna(value):
        return False
    return limits['min'] <= value <= limits['max']


def get_violations(df):
    if df is None:
        return None
    
    violation_data = []
    
    for metric, limits in STANDARD_LIMITS.items():
        cn_metric = REVERSE_MAPPING.get(metric, metric)
        if cn_metric not in df.columns:
            continue
        
        values = pd.to_numeric(df[cn_metric], errors='coerce')
        mask = (values < limits['min']) | (values > limits['max'])
        violations = df[mask].copy()
        
        if len(violations) > 0:
            for _, row in violations.iterrows():
                val = row[cn_metric]
                if pd.isna(val):
                    continue
                direction = '低于下限' if val < limits['min'] else '高于上限'
                magnitude = abs(val - (limits['min'] if val < limits['min'] else limits['max']))
                violation_data.append({
                    '时间': row['时间'],
                    '指标': cn_metric,
                    '数值': val,
                    '限值范围': f"{limits['min']}-{limits['max']} {limits['unit']}",
                    '超标类型': direction,
                    '超标幅度': round(magnitude, 4),
                })
    
    if violation_data:
        return pd.DataFrame(violation_data).sort_values('时间')
    return pd.DataFrame()


def get_monthly_violation_stats(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    violations_df = get_violations(df)
    if violations_df.empty:
        return pd.DataFrame()
    
    violations_df = violations_df.copy()
    violations_df['月份'] = pd.to_datetime(violations_df['时间']).dt.to_period('M')
    
    monthly_stats = violations_df.groupby(['月份', '指标']).agg(
        超标次数=('数值', 'count'),
        最大超标幅度=('超标幅度', 'max'),
    ).reset_index()
    
    total_samples = df.copy()
    total_samples['月份'] = pd.to_datetime(total_samples['时间']).dt.to_period('M')
    monthly_totals = total_samples.groupby('月份').size().reset_index(name='总样本数')
    
    monthly_stats = monthly_stats.merge(monthly_totals, on='月份', how='left')
    monthly_stats['超标率(%)'] = round(monthly_stats['超标次数'] / monthly_stats['总样本数'] * 100, 2)
    
    return monthly_stats


def get_consecutive_violations(df):
    if df is None or len(df) == 0:
        return {}
    
    result = {}
    
    for metric, limits in STANDARD_LIMITS.items():
        cn_metric = REVERSE_MAPPING.get(metric, metric)
        if cn_metric not in df.columns:
            continue
        
        values = pd.to_numeric(df[cn_metric], errors='coerce')
        mask = (values < limits['min']) | (values > limits['max'])
        
        max_consecutive = 0
        current_consecutive = 0
        for v in mask:
            if v:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        result[cn_metric] = max_consecutive
    
    return result


def detect_trend_warnings(df, window=3):
    if df is None or len(df) < window:
        return []
    
    warnings = []
    
    for metric, limits in STANDARD_LIMITS.items():
        cn_metric = REVERSE_MAPPING.get(metric, metric)
        if cn_metric not in df.columns:
            continue
        
        values = pd.to_numeric(df[cn_metric], errors='coerce').dropna()
        if len(values) < window:
            continue
        
        recent = values.tail(window).values
        
        if metric == 'effluent_turbidity' or metric == 'raw_ammonia':
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                current_val = recent[-1]
                threshold = limits['max']
                distance_to_limit = threshold - current_val
                if distance_to_limit < threshold * 0.3:
                    warnings.append({
                        '指标': cn_metric,
                        '趋势': '持续上升',
                        '当前值': round(current_val, 3),
                        '距离限值': round(distance_to_limit, 3),
                        '级别': '黄色预警',
                    })
        elif metric == 'effluent_chlorine':
            recent_vals = values.tail(window).values
            if all(recent_vals[i] >= recent_vals[i+1] for i in range(len(recent_vals)-1)):
                current_val = recent_vals[-1]
                distance_to_limit = current_val - limits['min']
                if distance_to_limit < limits['min'] * 0.5:
                    warnings.append({
                        '指标': cn_metric,
                        '趋势': '持续下降',
                        '当前值': round(current_val, 3),
                        '距离限值': round(distance_to_limit, 3),
                        '级别': '黄色预警',
                    })
    
    return warnings
