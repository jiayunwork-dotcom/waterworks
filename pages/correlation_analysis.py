import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.correlation_utils import (
    calculate_correlation,
    get_correlation_heatmap_data,
    cross_correlation,
    find_best_lag,
    detect_turbidity_shocks,
    analyze_shock_response,
)


st.header("🔬 工艺参数关联分析")

if st.session_state.df is None:
    st.info("请先上传数据或加载示例数据。")
else:
    df = st.session_state.df
    
    tab1, tab2, tab3 = st.tabs(["📊 相关性热力图", "⏱️ 滞后相关性分析", "⚡ 源水浊度突变检测"])
    
    with tab1:
        st.subheader("监测指标相关性热力图")
        
        corr_matrix = calculate_correlation(df)
        
        if corr_matrix.empty:
            st.warning("数据不足，无法计算相关性。")
        else:
            st.info("💡 颜色说明：绝对值>0.7 标红，0.4-0.7 标黄，<0.4 绿色")
            
            heatmap_data = get_correlation_heatmap_data(corr_matrix)
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                range_color=[-1, 1],
                title='Pearson相关系数矩阵',
                height=600,
            )
            
            fig.update_traces(
                hovertemplate='%{x}<br>%{y}<br>相关系数: %{z:.4f}<extra></extra>'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("强相关指标对 (|r| > 0.7)")
            
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.7:
                        strong_corr.append({
                            '指标1': corr_matrix.columns[i],
                            '指标2': corr_matrix.columns[j],
                            '相关系数': round(val, 4),
                            '强度': '极强正相关' if val > 0.9 else ('强正相关' if val > 0.7 else ('强负相关' if val < -0.7 else '极强负相关')),
                        })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True, hide_index=True)
            else:
                st.info("未发现强相关的指标对 (|r| > 0.7)。")
            
            st.divider()
            st.subheader("中等相关指标对 (0.4 < |r| ≤ 0.7)")
            
            moderate_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if 0.4 < abs(val) <= 0.7:
                        moderate_corr.append({
                            '指标1': corr_matrix.columns[i],
                            '指标2': corr_matrix.columns[j],
                            '相关系数': round(val, 4),
                            '方向': '正相关' if val > 0 else '负相关',
                        })
            
            if moderate_corr:
                st.dataframe(pd.DataFrame(moderate_corr), use_container_width=True, hide_index=True)
            else:
                st.info("未发现中等相关的指标对。")
    
    with tab2:
        st.subheader("投药量与沉后浊度的滞后相关性")
        
        st.info("分析混凝剂投加量变化后，沉后浊度需要多长时间才会响应。")
        
        if '混凝剂投加量' in df.columns and '沉后浊度' in df.columns:
            max_lag = st.slider("最大滞后步长 (小时)", 1, 48, 24)
            
            x_series = pd.to_numeric(df['混凝剂投加量'], errors='coerce')
            y_series = pd.to_numeric(df['沉后浊度'], errors='coerce')
            
            lags, corr_values = cross_correlation(x_series, y_series, max_lag=max_lag)
            
            best_lag, best_corr = find_best_lag(x_series, y_series, max_lag=max_lag)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最佳滞后步长", f"{best_lag} 小时")
            with col2:
                st.metric("最大相关系数", f"{best_corr:.4f}")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=lags,
                y=corr_values,
                name='互相关系数',
                marker_color='blue',
                opacity=0.7,
            ))
            
            fig.add_vline(x=best_lag, line_dash='dash', line_color='red', 
                         annotation_text=f'最佳滞后: {best_lag}h')
            
            fig.update_layout(
                title="投药量与沉后浊度互相关函数",
                xaxis_title='滞后时间 (小时)',
                yaxis_title='互相关系数',
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            💡 结果解读：
            - 最佳滞后时间为 **{best_lag} 小时**，即投药量变化后约 {best_lag} 小时沉后浊度响应最明显
            - 最大相关系数为 **{best_corr:.4f}**
            - 负的滞后值表示沉后浊度变化领先于投药量变化（可能是反馈控制）
            """)
            
            st.divider()
            st.subheader("散点图：投药量 vs 滞后沉后浊度")
            
            df_lagged = df.copy()
            df_lagged['沉后浊度_滞后'] = df_lagged['沉后浊度'].shift(-best_lag)
            df_lagged = df_lagged.dropna(subset=['混凝剂投加量', '沉后浊度_滞后'])
            
            fig = px.scatter(
                df_lagged,
                x='混凝剂投加量',
                y='沉后浊度_滞后',
                title=f'投药量 vs 滞后{best_lag}小时沉后浊度',
                trendline='ols',
                opacity=0.6,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("数据中缺少混凝剂投加量或沉后浊度列。")
    
    with tab3:
        st.subheader("源水浊度突变检测")
        
        threshold = st.slider("突变阈值 (变化率)", 0.1, 1.0, 0.5, 0.05,
                             help="相邻采样点浊度变化率超过此值即判定为突变")
        
        shocks_df = detect_turbidity_shocks(df, threshold=threshold)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("突变事件数", len(shocks_df) if not shocks_df.empty else 0)
        with col2:
            if not shocks_df.empty:
                avg_change = shocks_df['变化率(%)'].mean()
                st.metric("平均变化率", f"{avg_change:.2f}%")
            else:
                st.metric("平均变化率", "-")
        
        if not shocks_df.empty:
            st.dataframe(shocks_df, use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader("突变事件后投药量响应分析")
            
            response_window = st.slider("响应窗口 (小时)", 1, 12, 6)
            
            response_df = analyze_shock_response(
                df, shocks_df, 
                dosage_col='混凝剂投加量',
                response_window=response_window
            )
            
            if not response_df.empty:
                st.dataframe(response_df, use_container_width=True, hide_index=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    adequate = sum(response_df['响应是否充分'] == '是')
                    st.metric("响应充分的事件数", f"{adequate} / {len(response_df)}")
                with col2:
                    avg_adjust = response_df['调整幅度(%)'].mean()
                    st.metric("平均调整幅度", f"{avg_adjust:.2f}%")
                
                fig = px.scatter(
                    response_df,
                    x='浊度变化率(%)',
                    y='调整幅度(%)',
                    color='响应是否充分',
                    size='投药量调整量(mg/L)',
                    title='浊度突变幅度 vs 投药量调整幅度',
                    hover_data=['时间', '变化方向'],
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("无法进行响应分析（可能缺少投药量数据）。")
            
            st.divider()
            st.subheader("浊度时序图（突变事件标注）")
            
            turbidity = pd.to_numeric(df['源水浊度'], errors='coerce')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['时间'],
                y=turbidity,
                mode='lines',
                name='源水浊度',
                line=dict(color='blue', width=1.5),
            ))
            
            shock_times = shocks_df['时间'].values
            shock_values = [turbidity[df['时间'] == t].values[0] for t in shock_times 
                          if len(turbidity[df['时间'] == t]) > 0]
            
            if len(shock_times) > 0:
                fig.add_trace(go.Scatter(
                    x=shock_times,
                    y=shock_values,
                    mode='markers',
                    name='突变点',
                    marker=dict(color='red', size=10, symbol='circle'),
                ))
            
            fig.update_layout(
                title="源水浊度突变事件时序图",
                xaxis_title='时间',
                yaxis_title='源水浊度 (NTU)',
                height=450,
                hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ 未检测到源水浊度突变事件。")
