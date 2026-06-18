import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def first_order_decay(t, C0, k):
    return C0 * np.exp(-k * t)


def calculate_hydraulic_retention_time(volume_m3, flow_rate_m3h):
    if flow_rate_m3h <= 0:
        return 0
    return volume_m3 / flow_rate_m3h


def fit_decay_model(effluent_chlorine, retention_time_hours):
    if len(effluent_chlorine) < 2 or retention_time_hours <= 0:
        return None
    
    C0 = effluent_chlorine.iloc[0]
    if C0 <= 0:
        return None
    
    t_end = retention_time_hours
    C_end = effluent_chlorine.iloc[-1] if len(effluent_chlorine) > 1 else effluent_chlorine.iloc[0]
    
    if C_end <= 0:
        C_end = C0 * 0.5
    
    try:
        k_est = -np.log(C_end / C0) / t_end if t_end > 0 else 0.01
        
        t_data = np.linspace(0, t_end, len(effluent_chlorine))
        y_data = effluent_chlorine.values
        
        popt, pcov = curve_fit(
            first_order_decay, 
            t_data, 
            y_data, 
            p0=[C0, max(0.001, k_est)],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000,
        )
        
        return {
            'C0': popt[0],
            'k': popt[1],
            'C_end': first_order_decay(t_end, popt[0], popt[1]),
        }
    except Exception:
        k_est = 0.05
        return {
            'C0': C0,
            'k': k_est,
            'C_end': first_order_decay(t_end, C0, k_est),
        }


def calculate_monthly_decay(df, volume_m3, flow_rate_m3h):
    if df is None or '出水余氯' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df['月份'] = df['时间'].dt.month
    df['出水余氯'] = pd.to_numeric(df['出水余氯'], errors='coerce')
    
    results = []
    
    for month in sorted(df['月份'].unique()):
        month_data = df[df['月份'] == month]['出水余氯'].dropna()
        if len(month_data) < 3:
            continue
        
        fit_result = fit_decay_model(month_data, volume_m3 / flow_rate_m3h if flow_rate_m3h > 0 else 2)
        if fit_result:
            results.append({
                '月份': month,
                '衰减常数k (1/h)': round(fit_result['k'], 6),
                '初始余氯C0 (mg/L)': round(fit_result['C0'], 3),
                '末梢余氯 (mg/L)': round(fit_result['C_end'], 3),
                '样本数': len(month_data),
            })
    
    return pd.DataFrame(results)


def calculate_ct_value(effluent_chlorine_mg_l, contact_time_minutes):
    return effluent_chlorine_mg_l * contact_time_minutes


def check_ct_compliance(ct_value, standard=15.0):
    return {
        'ct_value': round(ct_value, 2),
        'standard': standard,
        'compliant': ct_value >= standard,
        'gap': round(standard - ct_value, 2) if ct_value < standard else 0,
    }


def get_retention_time_hours(volume_m3, flow_rate_m3h):
    if flow_rate_m3h <= 0:
        return 0
    return volume_m3 / flow_rate_m3h
