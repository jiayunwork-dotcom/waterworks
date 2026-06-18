import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chlorine_utils import (
    calculate_hydraulic_retention_time,
    fit_decay_model,
    calculate_monthly_decay,
    calculate_ct_value,
    check_ct_compliance,
    get_retention_time_hours,
    first_order_decay,
)


st.header("💧 余氯衰减与CT值计算")

if st.session_state.df is None:
    st.info("请先上传数据或加载示例数据。")
else:
    df = st.session_state.df
    
    tab1, tab2, tab3 = st.tabs(["CT值计算", "余氯衰减模型", "月度衰减常数"])
    
    with tab1:
        st.subheader("CT值计算与达标判定")
        
        col1, col2 = st.columns(2)
        with col1:
            pool_volume = st.number_input(
                "清水池容积 (m³)",
                min_value=100.0,
                value=2000.0,
                step=100.0,
                help="清水池的有效容积",
            )
        with col2:
            flow_rate = st.number_input(
                "供水流量 (m³/h)",
                min_value=10.0,
                value=500.0,
                step=10.0,
                help="平均供水流量",
            )
        
        retention_hours = calculate_hydraulic_retention_time(pool_volume, flow_rate)
        retention_minutes = retention_hours * 60
        
        st.metric("水力停留时间", f"{retention_hours:.2f} 小时 ({retention_minutes:.1f} 分钟)")
        
        st.divider()
        
        if '出水余氯' in df.columns:
            latest_chlorine = pd.to_numeric(df['出水余氯'], errors='coerce').dropna()
            if len(latest_chlorine) > 0:
                current_chlorine = latest_chlorine.iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("当前出厂余氯", f"{current_chlorine:.3f} mg/L")
                with col2:
                    ct_value = calculate_ct_value(current_chlorine, retention_minutes)
                    st.metric("当前CT值", f"{ct_value:.2f} mg·min/L")
                with col3:
                    ct_standard = st.number_input("CT值标准 (mg·min/L)", value=15.0, step=1.0)
                
                compliance = check_ct_compliance(ct_value, ct_standard)
                
                if compliance['compliant']:
                    st.success(f"✅ CT值达标 (标准: {ct_standard} mg·min/L)")
                else:
                    st.error(f"⚠️ CT值不达标！缺口: {compliance['gap']:.2f} mg·min/L")
                    st.warning("建议：提高加氯量或延长接触时间（如增大清水池容积、降低供水流量）")
        
        st.divider()
        
        st.subheader("CT值敏感性分析")
        
        chlorine_range = np.linspace(0.1, 5.0, 50)
        ct_values = [calculate_ct_value(c, retention_minutes) for c in chlorine_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chlorine_range,
            y=ct_values,
            mode='lines',
            name='CT值',
            line=dict(color='blue', width=2),
        ))
        fig.add_hline(
            y=ct_standard,
            line_dash='dash',
            line_color='red',
            annotation_text=f'标准: {ct_standard} mg·min/L',
        )
        if '出水余氯' in df.columns:
            latest_cl = pd.to_numeric(df['出水余氯'], errors='coerce').dropna()
            if len(latest_cl) > 0:
                curr_cl = latest_cl.iloc[-1]
                curr_ct = calculate_ct_value(curr_cl, retention_minutes)
                fig.add_trace(go.Scatter(
                    x=[curr_cl],
                    y=[curr_ct],
                    mode='markers',
                    name='当前值',
                    marker=dict(size=12, color='green'),
                ))
        
        fig.update_layout(
            title="余氯浓度与CT值关系",
            xaxis_title='出厂余氯 (mg/L)',
            yaxis_title='CT值 (mg·min/L)',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("余氯一阶衰减模型")
        
        st.latex(r'C(t) = C_0 \cdot e^{-kt}')
        st.caption("其中: C(t)为t时刻余氯浓度, C₀为初始余氯, k为衰减常数, t为时间")
        
        if '出水余氯' in df.columns:
            chlorine_data = pd.to_numeric(df['出水余氯'], errors='coerce').dropna()
            
            if len(chlorine_data) > 3:
                fit_result = fit_decay_model(chlorine_data, retention_hours)
                
                if fit_result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("初始余氯 C₀", f"{fit_result['C0']:.3f} mg/L")
                    with col2:
                        st.metric("衰减常数 k", f"{fit_result['k']:.6f} 1/h")
                    with col3:
                        st.metric("末梢余氯", f"{fit_result['C_end']:.3f} mg/L")
                    
                    t_range = np.linspace(0, retention_hours, 100)
                    c_decay = first_order_decay(t_range, fit_result['C0'], fit_result['k'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=t_range,
                        y=c_decay,
                        mode='lines',
                        name='拟合曲线',
                        line=dict(color='blue', width=2),
                    ))
                    fig.add_hline(
                        y=0.3,
                        line_dash='dash',
                        line_color='red',
                        annotation_text='管网末梢余氯下限 (0.3 mg/L)',
                    )
                    
                    fig.update_layout(
                        title="余氯衰减曲线",
                        xaxis_title='接触时间 (小时)',
                        yaxis_title='余氯浓度 (mg/L)',
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    half_life = np.log(2) / fit_result['k'] if fit_result['k'] > 0 else float('inf')
                    st.info(f"💡 余氯半衰期: {half_life:.2f} 小时")
                else:
                    st.warning("数据不足，无法拟合衰减模型。")
            else:
                st.warning("余氯数据点不足，无法拟合衰减模型。")
        else:
            st.info("数据中缺少出水余氯列。")
    
    with tab3:
        st.subheader("月度衰减常数变化")
        
        if '出水余氯' in df.columns:
            monthly_decay = calculate_monthly_decay(df, pool_volume, flow_rate)
            
            if not monthly_decay.empty:
                st.dataframe(monthly_decay, use_container_width=True, hide_index=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_decay['月份'],
                    y=monthly_decay['衰减常数k (1/h)'],
                    mode='lines+markers',
                    name='衰减常数k',
                    line=dict(color='red', width=2),
                    marker=dict(size=10),
                ))
                
                month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                               '7月', '8月', '9月', '10月', '11月', '12月']
                fig.update_xaxes(
                    tickvals=list(range(1, 13)),
                    ticktext=month_names,
                )
                
                fig.update_layout(
                    title="月份-衰减常数曲线",
                    xaxis_title='月份',
                    yaxis_title='衰减常数 k (1/h)',
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 通常温度越高，余氯衰减越快（k值越大）。夏季衰减常数通常高于冬季。")
            else:
                st.info("数据不足以计算月度衰减常数，确保数据覆盖多个月份。")
        else:
            st.info("数据中缺少出水余氯列。")
