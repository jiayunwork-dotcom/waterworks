import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trend_utils import (
    resample_data,
    stl_decomposition,
    get_seasonal_decomposition_df,
    flood_season_comparison,
    get_year_over_year,
    get_summary_stats_by_period,
)


st.header("📈 历史趋势与季节性分析")

if st.session_state.df is None:
    st.info("请先上传数据或加载示例数据。")
else:
    df = st.session_state.df
    
    core_metrics = ['源水浊度', '混凝剂投加量', '出水浊度', '出水余氯']
    available_metrics = [m for m in core_metrics if m in df.columns]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📉 长期趋势图", "🧩 STL季节性分解", "🌊 汛期vs非汛期", "📅 同比对比"])
    
    with tab1:
        st.subheader("核心指标长期趋势")
        
        selected_metrics = st.multiselect(
            "选择要展示的指标",
            available_metrics,
            default=available_metrics[:2] if available_metrics else [],
        )
        
        freq_options = {
            '原始': None,
            '按日': 'D',
            '按周': 'W',
            '按月': 'M',
        }
        freq_label = st.select_slider("数据聚合方式", options=list(freq_options.keys()), value='按日')
        freq = freq_options[freq_label]
        
        if selected_metrics:
            fig = go.Figure()
            
            for metric in selected_metrics:
                if freq:
                    series = resample_data(df, metric, freq=freq)
                else:
                    series = df.set_index('时间')[metric]
                
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name=metric,
                ))
            
            fig.update_layout(
                title=f"核心指标趋势图 ({freq_label}聚合)",
                xaxis_title='时间',
                yaxis_title='数值',
                height=500,
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            for metric in selected_metrics:
                st.subheader(f"{metric} 统计摘要")
                
                if freq:
                    stats = get_summary_stats_by_period(df, metric, period=freq.lower() if freq else 'day')
                else:
                    stats = get_summary_stats_by_period(df, metric, period='day')
                
                if not stats.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("均值", f"{stats['均值'].mean():.3f}")
                    with col2:
                        st.metric("最高值", f"{stats['最大值'].max():.3f}")
                    with col3:
                        st.metric("最低值", f"{stats['最小值'].min():.3f}")
                    with col4:
                        st.metric("标准差", f"{stats['均值'].std():.3f}")
        
        else:
            st.info("请选择至少一个指标进行展示。")
    
    with tab2:
        st.subheader("STL季节性分解")
        
        st.info("💡 STL分解将时序数据分解为趋势、季节和残差三个组分，帮助识别数据中的长期趋势和季节性波动。")
        
        stl_metric = st.selectbox(
            "选择要分解的指标",
            available_metrics,
            index=0 if available_metrics else None,
            key='stl_metric',
        )
        
        col1, col2 = st.columns(2)
        with col1:
            stl_freq = st.select_slider(
                "数据频率",
                options=['按日', '按周'],
                value='按日',
                key='stl_freq',
            )
        with col2:
            period = st.slider(
                "季节周期",
                7, 365, 30,
                help="季节性成分的周期长度",
                key='stl_period',
            )
        
        if stl_metric:
            freq_param = 'D' if stl_freq == '按日' else 'W'
            series = resample_data(df, stl_metric, freq=freq_param)
            
            if series is not None and len(series) >= 2 * period:
                stl_result = stl_decomposition(series, period=period)
                
                if stl_result is not None:
                    decomp_df = get_seasonal_decomposition_df(stl_result)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=decomp_df.index,
                        y=decomp_df['原始'],
                        mode='lines',
                        name='原始数据',
                        line=dict(color='blue'),
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=decomp_df.index,
                        y=decomp_df['趋势'],
                        mode='lines',
                        name='趋势成分',
                        line=dict(color='red', width=2),
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=decomp_df.index,
                        y=decomp_df['季节'],
                        mode='lines',
                        name='季节成分',
                        line=dict(color='green'),
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=decomp_df.index,
                        y=decomp_df['残差'],
                        mode='lines',
                        name='残差成分',
                        line=dict(color='gray'),
                    ))
                    
                    fig.update_layout(
                        title=f"{stl_metric} STL季节分解",
                        xaxis_title='时间',
                        yaxis_title='数值',
                        height=550,
                        hovermode='x unified',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.subheader("各成分统计")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("趋势强度", f"{stl_result.trend.std() / stl_result.observed.std() * 100:.1f}%")
                    with col2:
                        st.metric("季节强度", f"{stl_result.seasonal.std() / stl_result.observed.std() * 100:.1f}%")
                    with col3:
                        st.metric("残差强度", f"{stl_result.resid.std() / stl_result.observed.std() * 100:.1f}%")
                    with col4:
                        st.metric("数据点数", len(series))
                    
                    seasonal_strength = stl_result.seasonal.std() / stl_result.observed.std()
                    if seasonal_strength > 0.3:
                        st.success(f"✅ 该指标存在明显的季节性波动（季节强度: {seasonal_strength:.1%}）")
                    elif seasonal_strength > 0.1:
                        st.warning(f"⚠️ 该指标存在一定的季节性波动（季节强度: {seasonal_strength:.1%}）")
                    else:
                        st.info(f"ℹ️ 该指标季节性不明显（季节强度: {seasonal_strength:.1%}）")
                else:
                    st.warning("STL分解失败，请尝试调整参数。")
            else:
                st.warning(f"数据点不足，至少需要 {2*period} 个数据点。当前: {len(series) if series is not None else 0} 个点")
    
    with tab3:
        st.subheader("汛期 vs 非汛期对比")
        
        st.info("💡 汛期通常为6-9月，此时源水浊度较高，混凝剂投药量可能需要相应调整。")
        
        flood_metric = st.selectbox(
            "选择对比指标",
            available_metrics,
            index=1 if len(available_metrics) > 1 else 0,
            key='flood_metric',
        )
        
        flood_months = st.multiselect(
            "汛期月份",
            list(range(1, 13)),
            default=[6, 7, 8, 9],
        )
        
        if flood_metric and flood_months:
            comparison_df = flood_season_comparison(df, flood_metric, tuple(flood_months))
            
            if not comparison_df.empty:
                fig = px.box(
                    comparison_df,
                    x='时期',
                    y=flood_metric,
                    color='时期',
                    title=f"{flood_metric} 汛期 vs 非汛期箱线图",
                    points='outliers',
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("统计对比")
                
                flood_data = comparison_df[comparison_df['时期'] == '汛期'][flood_metric]
                non_flood_data = comparison_df[comparison_df['时期'] == '非汛期'][flood_metric]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**汛期**")
                    st.metric("样本数", len(flood_data))
                    st.metric("均值", f"{flood_data.mean():.3f}")
                    st.metric("中位数", f"{flood_data.median():.3f}")
                    st.metric("标准差", f"{flood_data.std():.3f}")
                    st.metric("最大值", f"{flood_data.max():.3f}")
                    st.metric("最小值", f"{flood_data.min():.3f}")
                
                with col2:
                    st.markdown("**非汛期**")
                    st.metric("样本数", len(non_flood_data))
                    st.metric("均值", f"{non_flood_data.mean():.3f}")
                    st.metric("中位数", f"{non_flood_data.median():.3f}")
                    st.metric("标准差", f"{non_flood_data.std():.3f}")
                    st.metric("最大值", f"{non_flood_data.max():.3f}")
                    st.metric("最小值", f"{non_flood_data.min():.3f}")
                
                if len(flood_data) > 0 and len(non_flood_data) > 0:
                    diff_pct = (flood_data.mean() - non_flood_data.mean()) / non_flood_data.mean() * 100
                    st.info(f"""
                    💡 分析结论：
                    - 汛期{ flood_metric }均值比非汛期{'高' if diff_pct > 0 else '低'} {abs(diff_pct):.1f}%
                    - 汛期波动{'更大' if flood_data.std() > non_flood_data.std() else '更小'}
                    """)
    
    with tab4:
        st.subheader("同比对比分析")
        
        st.info("💡 对比同一指标在不同年份的变化，帮助识别年度趋势。")
        
        yoy_metric = st.selectbox(
            "选择对比指标",
            available_metrics,
            index=0 if available_metrics else None,
            key='yoy_metric',
        )
        
        if yoy_metric:
            yoy_df = get_year_over_year(df, yoy_metric, freq='D')
            
            if not yoy_df.empty and len(yoy_df.columns) >= 1:
                years = list(yoy_df.columns)
                
                fig = go.Figure()
                
                for i, year in enumerate(years):
                    line_style = 'solid' if i == len(years) - 1 else 'dash'
                    fig.add_trace(go.Scatter(
                        x=yoy_df.index,
                        y=yoy_df[year],
                        mode='lines',
                        name=f'{year}年',
                        line=dict(dash=line_style if i < len(years) - 1 else None),
                    ))
                
                fig.update_layout(
                    title=f"{yoy_metric} 年度同比对比",
                    xaxis_title='一年中的第几天',
                    yaxis_title=yoy_metric,
                    height=450,
                    hovermode='x unified',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("年度统计对比")
                
                annual_stats = []
                for year in years:
                    data = yoy_df[year].dropna()
                    if len(data) > 0:
                        annual_stats.append({
                            '年份': year,
                            '均值': data.mean(),
                            '最大值': data.max(),
                            '最小值': data.min(),
                            '标准差': data.std(),
                            '数据天数': len(data),
                        })
                
                if annual_stats:
                    stats_df = pd.DataFrame(annual_stats)
                    for col in ['均值', '最大值', '最小值', '标准差']:
                        stats_df[col] = stats_df[col].round(3)
                    
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    if len(annual_stats) >= 2:
                        latest = annual_stats[-1]
                        previous = annual_stats[-2]
                        change = (latest['均值'] - previous['均值']) / previous['均值'] * 100
                        
                        if abs(change) > 5:
                            direction = '显著上升' if change > 0 else '显著下降'
                            st.warning(f"⚠️ 相比去年，{yoy_metric}年均值{direction} {abs(change):.1f}%")
                        else:
                            st.success(f"✅ 相比去年，{yoy_metric}年均值变化不大（{change:+.1f}%）")
            else:
                st.warning("数据不足以进行同比对比，需要至少包含多个年份的数据。")
