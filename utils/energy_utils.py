import pandas as pd
import numpy as np


ENERGY_COLS_CN = ['取水泵(kWh)', '搅拌器(kWh)', '反冲洗泵(kWh)', '加压泵(kWh)']
FLOW_COL_CN = '供水量(m³)'


def load_energy_data(file):
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期')
        return df
    except Exception as e:
        return None


def calculate_unit_energy(df):
    if df is None or FLOW_COL_CN not in df.columns:
        return None, None
    
    df = df.copy()
    
    energy_cols = [c for c in ENERGY_COLS_CN if c in df.columns]
    
    df['总能耗(kWh)'] = df[energy_cols].sum(axis=1)
    df['单位水量能耗(kWh/m³)'] = df['总能耗(kWh)'] / df[FLOW_COL_CN].replace(0, np.nan)
    
    return df, energy_cols


def get_energy_breakdown(df, energy_cols):
    if df is None or not energy_cols:
        return pd.DataFrame()
    
    totals = {col: df[col].sum() for col in energy_cols}
    total_all = sum(totals.values())
    
    breakdown = []
    for col, val in totals.items():
        breakdown.append({
            '环节': col.replace('(kWh)', ''),
            '能耗(kWh)': round(val, 2),
            '占比(%)': round(val / total_all * 100, 2) if total_all > 0 else 0,
        })
    
    return pd.DataFrame(breakdown)


def compare_periods(df, start_date1, end_date1, start_date2, end_date2, energy_cols):
    if df is None or '日期' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    
    period1 = df[(df['日期'] >= pd.to_datetime(start_date1)) & (df['日期'] <= pd.to_datetime(end_date1))]
    period2 = df[(df['日期'] >= pd.to_datetime(start_date2)) & (df['日期'] <= pd.to_datetime(end_date2))]
    
    if len(period1) == 0 or len(period2) == 0:
        return pd.DataFrame()
    
    comparison = []
    for col in energy_cols:
        val1 = period1[col].mean()
        val2 = period2[col].mean()
        change = val2 - val1
        change_pct = (change / val1 * 100) if val1 != 0 else 0
        
        comparison.append({
            '环节': col.replace('(kWh)', ''),
            '阶段1日均(kWh)': round(val1, 2),
            '阶段2日均(kWh)': round(val2, 2),
            '变化量(kWh)': round(change, 2),
            '变化率(%)': round(change_pct, 2),
        })
    
    return pd.DataFrame(comparison)


def detect_energy_anomalies(df, energy_cols, window_days=30, n_std=2):
    if df is None or len(df) < window_days:
        return pd.DataFrame()
    
    df = df.copy()
    df = df.sort_values('日期')
    
    anomalies = []
    
    for col in energy_cols:
        values = df[col].values
        
        for i in range(window_days, len(values)):
            window = values[i-window_days:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std == 0:
                continue
            
            deviation = abs(values[i] - mean) / std
            
            if deviation > n_std:
                anomalies.append({
                    '日期': df.iloc[i]['日期'],
                    '环节': col.replace('(kWh)', ''),
                    '能耗值(kWh)': round(values[i], 2),
                    '30天均值(kWh)': round(mean, 2),
                    '偏离标准差': round(deviation, 2),
                    '偏离方向': '偏高' if values[i] > mean else '偏低',
                })
    
    if anomalies:
        return pd.DataFrame(anomalies).sort_values('日期', ascending=False)
    return pd.DataFrame()


def get_energy_summary(df, energy_cols):
    if df is None or not energy_cols:
        return {}
    
    summary = {}
    
    for col in energy_cols:
        summary[col.replace('(kWh)', '')] = {
            '累计': round(df[col].sum(), 2),
            '日均': round(df[col].mean(), 2),
            '最高': round(df[col].max(), 2),
            '最低': round(df[col].min(), 2),
        }
    
    if '总能耗(kWh)' in df.columns:
        summary['总计'] = {
            '累计': round(df['总能耗(kWh)'].sum(), 2),
            '日均': round(df['总能耗(kWh)'].mean(), 2),
            '最高': round(df['总能耗(kWh)'].max(), 2),
            '最低': round(df['总能耗(kWh)'].min(), 2),
        }
    
    if '单位水量能耗(kWh/m³)' in df.columns:
        summary['单位水量能耗'] = {
            '均值': round(df['单位水量能耗(kWh/m³)'].mean(), 4),
            '最高': round(df['单位水量能耗(kWh/m³)'].max(), 4),
            '最低': round(df['单位水量能耗(kWh/m³)'].min(), 4),
        }
    
    return summary
