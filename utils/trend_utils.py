import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


def resample_data(df, value_col, freq='D'):
    if df is None or value_col not in df.columns:
        return None
    
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    resampled = df.set_index('时间')[value_col].resample(freq).mean()
    return resampled


def stl_decomposition(series, period=None):
    if series is None or len(series) < 2 * (period or 7):
        return None
    
    series = series.dropna()
    if len(series) < 14:
        return None
    
    if period is None:
        period = 7
    
    try:
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        return result
    except Exception:
        return None


def get_seasonal_decomposition_df(result):
    if result is None:
        return pd.DataFrame()
    
    decomp_df = pd.DataFrame({
        '趋势': result.trend,
        '季节': result.seasonal,
        '残差': result.resid,
        '原始': result.observed,
    })
    
    return decomp_df


def flood_season_comparison(df, value_col, flood_months=(6, 7, 8, 9)):
    if df is None or value_col not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df['月份'] = df['时间'].dt.month
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    
    df['时期'] = df['月份'].apply(
        lambda x: '汛期' if x in flood_months else '非汛期'
    )
    
    return df[['时期', value_col]]


def get_year_over_year(df, value_col, freq='D'):
    if df is None or value_col not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    resampled = df.set_index('时间')[value_col].resample(freq).mean().reset_index()
    resampled['year'] = resampled['时间'].dt.year
    resampled['day_of_year'] = resampled['时间'].dt.dayofyear
    
    pivot = resampled.pivot(index='day_of_year', columns='year', values=value_col)
    
    return pivot


def get_summary_stats_by_period(df, value_col, period='month'):
    if df is None or value_col not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    if period == 'month':
        df['period'] = df['时间'].dt.to_period('M')
    elif period == 'week':
        df['period'] = df['时间'].dt.to_period('W')
    elif period == 'day':
        df['period'] = df['时间'].dt.date
    else:
        df['period'] = df['时间'].dt.to_period('M')
    
    stats = df.groupby('period')[value_col].agg([
        ('均值', 'mean'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('最大值', 'max'),
        ('样本数', 'count'),
    ]).reset_index()
    
    stats['period'] = stats['period'].astype(str)
    
    return stats
