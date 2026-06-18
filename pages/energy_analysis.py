import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.energy_utils import (
    load_energy_data,
    calculate_unit_energy,
    get_energy_breakdown,
    compare_periods,
    detect_energy_anomalies,
    get_energy_summary,
    ENERGY_COLS_CN,
)


st.header("⚡ 能耗分析")

if st.session_state.df is None and st.session_state.energy_df is None:
    st.info("请先上传能耗数据或加载示例数据。")
else:
    energy_df = st.session_state.energy_df
    
    if energy_df is None:
        st.info("请在左侧边栏上传能耗数据CSV文件。")
    else:
        energy_df_processed, energy_cols = calculate_unit_energy(energy_df)
        breakdown_df = get_energy_breakdown(energy_df_processed, energy_cols)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 能耗概览", "🥧 能耗构成", "📈 同比环比分析", "⚠️ 能耗异常检测"])
        
        with tab1:
            st.subheader("能耗概览")
            
            summary = get_energy_summary(energy_df_processed, energy_cols)
            
            if summary:
                cols = st.columns(len(summary))
                for i, (name, stats) in enumerate(summary.items()):
                    with cols[i]:
                        if '累计' in stats:
                            st.metric(name, f"{stats.get('累计', stats.get('均值', 0)):,.2f}")
                        else:
                            st.metric(name, f"{stats.get('均值', 0):.4f}")
            
            st.divider()
            st.subheader("能耗时序图")
            
            energy_type = st.multiselect(
                "选择展示的能耗类型",
                [c.replace('(kWh)', '') for c in energy_cols],
                default=[c.replace('(kWh)', '') for c in energy_cols],
            )
            
            fig = go.Figure()
            for col in energy_cols:
                short_name = col.replace('(kWh)', '')
                if short_name in energy_type:
                    fig.add_trace(go.Scatter(
                        x=energy_df_processed['日期'],
                        y=energy_df_processed[col],
                        mode='lines',
                        name=short_name,
                    ))
            
            fig.update_layout(
                title="各环节能耗变化趋势",
                xaxis_title='日期',
                yaxis_title='能耗 (kWh)',
                height=450,
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("单位水量能耗")
            
            if '单位水量能耗(kWh/m³)' in energy_df_processed.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=energy_df_processed['日期'],
                    y=energy_df_processed['单位水量能耗(kWh/m³)'],
                    mode='lines+markers',
                    name='单位水量能耗',
                    line=dict(color='green'),
                ))
                
                avg_unit = energy_df_processed['单位水量能耗(kWh/m³)'].mean()
                fig.add_hline(
                    y=avg_unit,
                    line_dash='dash',
                    line_color='red',
                    annotation_text=f'均值: {avg_unit:.4f} kWh/m³',
                )
                
                fig.update_layout(
                    title="单位水量能耗变化",
                    xaxis_title='日期',
                    yaxis_title='单位水量能耗 (kWh/m³)',
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均单位能耗", f"{avg_unit:.4f} kWh/m³")
                with col2:
                    st.metric("最高单位能耗", f"{energy_df_processed['单位水量能耗(kWh/m³)'].max():.4f} kWh/m³")
                with col3:
                    st.metric("最低单位能耗", f"{energy_df_processed['单位水量能耗(kWh/m³)'].min():.4f} kWh/m³")
        
        with tab2:
            st.subheader("能耗构成分析")
            
            if not breakdown_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        breakdown_df,
                        values='能耗(kWh)',
                        names='环节',
                        title='各工艺环节能耗占比',
                        hole=0.4,
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    
                    total_energy = breakdown_df['能耗(kWh)'].sum()
                    st.metric("总能耗", f"{total_energy:,.2f} kWh")
                
                st.divider()
                st.subheader("累计能耗趋势")
                
                fig = go.Figure()
                for col in energy_cols:
                    short_name = col.replace('(kWh)', '')
                    cumulative = energy_df_processed[col].cumsum()
                    fig.add_trace(go.Scatter(
                        x=energy_df_processed['日期'],
                        y=cumulative,
                        mode='lines',
                        name=short_name,
                        stackgroup='one',
                    ))
                
                fig.update_layout(
                    title="各环节累计能耗 (堆叠面积图)",
                    xaxis_title='日期',
                    yaxis_title='累计能耗 (kWh)',
                    height=450,
                    hovermode='x unified',
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("数据不足，无法分析能耗构成。")
        
        with tab3:
            st.subheader("同比环比分析")
            
            if len(energy_df_processed) < 14:
                st.warning("数据量不足，建议至少有14天数据进行对比分析。")
            else:
                col1, col2 = st.columns(2)
                
                min_date = energy_df_processed['日期'].min().date()
                max_date = energy_df_processed['日期'].max().date()
                mid_date = min_date + (max_date - min_date) / 2
                
                with col1:
                    st.markdown("**阶段1 (基准)**")
                    start1 = st.date_input("开始日期", value=min_date, key='start1')
                    end1 = st.date_input("结束日期", value=mid_date, key='end1')
                
                with col2:
                    st.markdown("**阶段2 (对比)**")
                    start2 = st.date_input("开始日期", value=mid_date, key='start2')
                    end2 = st.date_input("结束日期", value=max_date, key='end2')
                
                comparison_df = compare_periods(
                    energy_df_processed,
                    start1, end1,
                    start2, end2,
                    energy_cols,
                )
                
                if not comparison_df.empty:
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['环节'],
                        y=comparison_df['阶段1日均(kWh)'],
                        name='阶段1日均',
                        marker_color='lightblue',
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['环节'],
                        y=comparison_df['阶段2日均(kWh)'],
                        name='阶段2日均',
                        marker_color='orange',
                    ))
                    
                    fig.update_layout(
                        title="两阶段日均能耗对比",
                        xaxis_title='工艺环节',
                        yaxis_title='日均能耗 (kWh)',
                        barmode='group',
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.subheader("变化率")
                    
                    fig = go.Figure()
                    colors = ['red' if x > 0 else 'green' for x in comparison_df['变化率(%)']]
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['环节'],
                        y=comparison_df['变化率(%)'],
                        text=comparison_df['变化率(%)'].apply(lambda x: f'{x:+.2f}%'),
                        textposition='auto',
                        marker_color=colors,
                    ))
                    
                    fig.update_layout(
                        title="各环节能耗变化率 (阶段2 vs 阶段1)",
                        xaxis_title='工艺环节',
                        yaxis_title='变化率 (%)',
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("无法进行对比分析，请检查日期选择。")
        
        with tab4:
            st.subheader("能耗异常检测")
            
            st.info("📢 检测规则：某日能耗偏离近30天均值超过2个标准差则标记为异常。")
            
            col1, col2 = st.columns(2)
            with col1:
                window_days = st.slider("参考窗口 (天)", 7, 60, 30)
            with col2:
                n_std = st.slider("标准差阈值", 1.0, 3.0, 2.0, 0.5)
            
            anomalies_df = detect_energy_anomalies(
                energy_df_processed,
                energy_cols,
                window_days=window_days,
                n_std=n_std,
            )
            
            if not anomalies_df.empty:
                st.metric("异常事件数", len(anomalies_df))
                
                st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
                
                st.divider()
                st.subheader("异常分布")
                
                fig = px.histogram(
                    anomalies_df,
                    x='环节',
                    color='偏离方向',
                    title='各环节异常事件分布',
                    barmode='group',
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("异常值标注时序图")
                
                selected_anomaly_col = st.selectbox(
                    "选择查看的环节",
                    [c.replace('(kWh)', '') for c in energy_cols],
                )
                
                col_name = f"{selected_anomaly_col}(kWh)"
                if col_name in energy_df_processed.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=energy_df_processed['日期'],
                        y=energy_df_processed[col_name],
                        mode='lines',
                        name=selected_anomaly_col,
                        line=dict(color='blue'),
                    ))
                    
                    col_anomalies = anomalies_df[anomalies_df['环节'] == selected_anomaly_col]
                    if not col_anomalies.empty:
                        fig.add_trace(go.Scatter(
                            x=col_anomalies['日期'],
                            y=col_anomalies['能耗值(kWh)'],
                            mode='markers',
                            name='异常点',
                            marker=dict(color='red', size=10, symbol='circle'),
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_anomaly_col} 能耗异常检测",
                        xaxis_title='日期',
                        yaxis_title='能耗 (kWh)',
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ 未检测到能耗异常事件！")
