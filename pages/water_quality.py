import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import (
    get_violations,
    get_monthly_violation_stats,
    get_consecutive_violations,
    detect_trend_warnings,
    STANDARD_LIMITS,
    REVERSE_MAPPING,
    check_standard,
)


st.header("📋 水质达标监控与超标预警")

if st.session_state.df is None:
    st.info("请先上传数据或加载示例数据。")
else:
    df = st.session_state.df
    
    violations_df = get_violations(df)
    consecutive_violations = get_consecutive_violations(df)
    trend_warnings = detect_trend_warnings(df)
    
    st.subheader("国标限值配置")
    
    with st.expander("⚙️ 国标限值设置", expanded=False):
        col1, col2 = st.columns(2)
        
        limits = {}
        for i, (metric_en, limit_info) in enumerate(STANDARD_LIMITS.items()):
            metric_cn = REVERSE_MAPPING.get(metric_en, metric_en)
            if i % 2 == 0:
                with col1:
                    st.markdown(f"**{metric_cn}**")
                    min_val = st.number_input(
                        f"下限 ({limit_info['unit']})",
                        value=float(limit_info['min']),
                        key=f"min_{metric_en}",
                    )
                    max_val = st.number_input(
                        f"上限 ({limit_info['unit']})",
                        value=float(limit_info['max']),
                        key=f"max_{metric_en}",
                    )
                    limits[metric_en] = {'min': min_val, 'max': max_val, 'unit': limit_info['unit']}
            else:
                with col2:
                    st.markdown(f"**{metric_cn}**")
                    min_val = st.number_input(
                        f"下限 ({limit_info['unit']})",
                        value=float(limit_info['min']),
                        key=f"min_{metric_en}",
                    )
                    max_val = st.number_input(
                        f"上限 ({limit_info['unit']})",
                        value=float(limit_info['max']),
                        key=f"max_{metric_en}",
                    )
                    limits[metric_en] = {'min': min_val, 'max': max_val, 'unit': limit_info['unit']}
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 达标时序图", "📊 超标统计", "⚠️ 超标事件列表", "📉 趋势预警"])
    
    with tab1:
        st.subheader("水质指标时序图（超标点红色标注）")
        
        metric_options = []
        for metric_en in STANDARD_LIMITS.keys():
            metric_cn = REVERSE_MAPPING.get(metric_en, metric_en)
            if metric_cn in df.columns:
                metric_options.append(metric_cn)
        
        selected_metric = st.selectbox("选择指标", metric_options, index=0 if metric_options else None)
        
        if selected_metric:
            metric_en = None
            for k, v in REVERSE_MAPPING.items():
                if v == selected_metric:
                    metric_en = k
                    break
            
            limit_info = STANDARD_LIMITS.get(metric_en, {'min': 0, 'max': 1, 'unit': ''})
            
            values = pd.to_numeric(df[selected_metric], errors='coerce')
            mask = (values < limit_info['min']) | (values > limit_info['max'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['时间'],
                y=values,
                mode='lines',
                name=selected_metric,
                line=dict(color='blue', width=1.5),
            ))
            
            violation_times = df.loc[mask, '时间']
            violation_values = values[mask]
            fig.add_trace(go.Scatter(
                x=violation_times,
                y=violation_values,
                mode='markers',
                name='超标点',
                marker=dict(color='red', size=8, symbol='circle'),
            ))
            
            fig.add_hline(
                y=limit_info['max'],
                line_dash='dash',
                line_color='red',
                annotation_text=f"上限: {limit_info['max']} {limit_info['unit']}",
            )
            if limit_info['min'] > 0:
                fig.add_hline(
                    y=limit_info['min'],
                    line_dash='dash',
                    line_color='orange',
                    annotation_text=f"下限: {limit_info['min']} {limit_info['unit']}",
                )
            
            fig.update_layout(
                title=f"{selected_metric} 时序图",
                xaxis_title='时间',
                yaxis_title=f"{selected_metric} ({limit_info['unit']})",
                height=500,
                hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total_count = len(values.dropna())
                st.metric("总样本数", total_count)
            with col2:
                violation_count = mask.sum()
                st.metric("超标次数", violation_count)
            with col3:
                violation_rate = violation_count / total_count * 100 if total_count > 0 else 0
                st.metric("超标率", f"{violation_rate:.2f}%")
    
    with tab2:
        st.subheader("月度超标统计")
        
        monthly_stats = get_monthly_violation_stats(df)
        
        if not monthly_stats.empty:
            st.dataframe(monthly_stats, use_container_width=True, hide_index=True)
            
            fig = px.bar(
                monthly_stats,
                x='月份',
                y='超标次数',
                color='指标',
                title='各指标月度超标次数',
                barmode='group',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ 暂无超标记录")
        
        st.divider()
        st.subheader("连续超标最长时段")
        
        if consecutive_violations:
            cols = st.columns(len(consecutive_violations))
            for i, (metric, max_consec) in enumerate(consecutive_violations.items()):
                with cols[i % len(cols)]:
                    st.metric(metric, f"{max_consec} 个采样点")
        else:
            st.info("暂无连续超标数据")
    
    with tab3:
        st.subheader("超标事件列表")
        
        if violations_df is not None and not violations_df.empty:
            st.dataframe(violations_df, use_container_width=True, hide_index=True)
            
            st.download_button(
                "📥 下载超标记录 (CSV)",
                violations_df.to_csv(index=False).encode('utf-8-sig'),
                "超标记录.csv",
                "text/csv",
            )
        else:
            st.success("✅ 暂无超标记录，所有指标均达标！")
    
    with tab4:
        st.subheader("趋势预警")
        
        st.info("📢 预警规则：连续3个采样点某指标呈单调恶化趋势（即使尚未超标），触发黄色趋势预警。")
        
        if trend_warnings:
            for w in trend_warnings:
                with st.container():
                    st.warning(f"⚠️ {w['级别']}: {w['指标']} 呈{w['趋势']}趋势")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("当前值", w['当前值'])
                    with col2:
                        st.metric("距离限值", w['距离限值'])
        else:
            st.success("✅ 暂无趋势预警，所有指标趋势平稳。")
        
        st.divider()
        st.subheader("指标变化趋势分析")
        
        trend_metric = st.selectbox(
            "选择要分析趋势的指标",
            metric_options,
            index=0 if metric_options else None,
            key="trend_metric",
        )
        
        if trend_metric:
            window_size = st.slider("滑动窗口大小", 3, 24, 5)
            
            values = pd.to_numeric(df[trend_metric], errors='coerce').dropna()
            
            rolling_mean = values.rolling(window=window_size).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['时间'],
                y=values,
                mode='lines',
                name='原始数据',
                line=dict(color='lightblue', width=1),
                opacity=0.6,
            ))
            fig.add_trace(go.Scatter(
                x=df['时间'],
                y=rolling_mean,
                mode='lines',
                name=f'{window_size}点滑动平均',
                line=dict(color='red', width=2),
            ))
            
            fig.update_layout(
                title=f"{trend_metric} 趋势分析",
                xaxis_title='时间',
                yaxis_title=trend_metric,
                height=450,
                hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True)
