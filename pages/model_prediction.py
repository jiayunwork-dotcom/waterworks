import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (
    prepare_training_data,
    train_model,
    compare_models,
    predict_dosage,
    save_model,
    load_model_versions,
    FEATURE_COLS_CN,
    TARGET_COL_CN,
    MODEL_TYPES,
)


st.header("🧪 投药量预测模型")

if st.session_state.df is None:
    st.info("请先上传数据或加载示例数据。")
else:
    df = st.session_state.df
    
    X, y = prepare_training_data(df)
    
    if X is None or y is None:
        st.warning("⚠️ 数据不足，无法训练模型。请确保数据包含必要的特征列和目标列，且有效数据点不少于20个。")
        st.write("所需特征列：", FEATURE_COLS_CN)
        st.write("目标列：", TARGET_COL_CN)
    else:
        st.success(f"✅ 可用训练数据: {len(X)} 条记录，{len(X.columns)} 个特征")
        
        tab1, tab2, tab3, tab4 = st.tabs(["模型训练与对比", "投药量预测", "模型版本管理", "模型详情"])
        
        with tab1:
            st.subheader("模型训练与对比")
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
            with col2:
                random_state = st.number_input("随机种子", value=42, min_value=1, max_value=9999)
            
            if st.button("🚀 训练所有模型并对比", type="primary", use_container_width=True):
                with st.spinner("正在训练模型..."):
                    comparison_df, results = compare_models(X, y)
                    st.session_state.comparison_df = comparison_df
                    st.session_state.model_results = results
            
            if 'comparison_df' in st.session_state and st.session_state.comparison_df is not None:
                st.subheader("模型性能对比")
                st.dataframe(st.session_state.comparison_df, use_container_width=True, hide_index=True)
                
                fig = go.Figure()
                
                for _, row in st.session_state.comparison_df.iterrows():
                    model_name = row['模型']
                    fig.add_trace(go.Bar(
                        name=model_name,
                        x=['训练集R²', '测试集R²', '训练集RMSE', '测试集RMSE', '训练集MAE', '测试集MAE'],
                        y=[row['训练集R²'], row['测试集R²'], row['训练集RMSE'], row['测试集RMSE'], row['训练集MAE'], row['测试集MAE']],
                        text=[f"{row['训练集R²']:.4f}", f"{row['测试集R²']:.4f}", 
                              f"{row['训练集RMSE']:.4f}", f"{row['测试集RMSE']:.4f}",
                              f"{row['训练集MAE']:.4f}", f"{row['测试集MAE']:.4f}"],
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title="模型性能指标对比",
                    barmode='group',
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("投药量预测")
            
            model_options = list(MODEL_TYPES.keys())
            selected_model_name = st.selectbox("选择模型", model_options, index=1)
            selected_model_type = MODEL_TYPES[selected_model_name]
            
            if st.session_state.model_results is not None and selected_model_name in st.session_state.model_results:
                model_result = st.session_state.model_results[selected_model_name]
                st.session_state.selected_model = model_result
                
                st.info(f"当前模型: {selected_model_name}，测试集R²: {model_result['test_metrics']['R²']:.4f}")
                
                st.subheader("输入源水参数")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    turbidity = st.number_input("源水浊度 (NTU)", min_value=0.0, value=10.0, step=0.1)
                    ph = st.number_input("源水pH", min_value=0.0, max_value=14.0, value=7.0, step=0.01)
                with col2:
                    temperature = st.number_input("源水温度 (℃)", min_value=0.0, value=15.0, step=0.1)
                    ammonia = st.number_input("源水氨氮 (mg/L)", min_value=0.0, value=0.5, step=0.01)
                with col3:
                    uv254 = st.number_input("源水有机物 (UV254)", min_value=0.0, value=0.1, step=0.001)
                
                confidence = st.select_slider("置信度", options=[0.9, 0.95, 0.99], value=0.95)
                
                features = {
                    '源水浊度': turbidity,
                    '源水pH': ph,
                    '源水温度': temperature,
                    '源水氨氮': ammonia,
                    '源水有机物': uv254,
                }
                
                if st.button("🔮 预测投药量", type="primary"):
                    prediction_result = predict_dosage(model_result, features, confidence=confidence)
                    
                    st.subheader("预测结果")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("推荐投药量", f"{prediction_result['prediction']} mg/L")
                    with col2:
                        st.metric(f"置信区间下限 ({int(confidence*100)}%)", f"{prediction_result['lower']} mg/L")
                    with col3:
                        st.metric(f"置信区间上限 ({int(confidence*100)}%)", f"{prediction_result['upper']} mg/L")
                    
                    st.success(f"💡 建议混凝剂投加量范围: {prediction_result['lower']} ~ {prediction_result['upper']} mg/L")
            else:
                st.info("请先在'模型训练与对比'标签页中训练模型。")
        
        with tab3:
            st.subheader("模型版本管理")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("保留版本数", "最近3个版本")
            with col2:
                if st.button("💾 保存当前模型", disabled=st.session_state.selected_model is None):
                    if st.session_state.selected_model is not None:
                        filepath, timestamp = save_model(st.session_state.selected_model)
                        st.success(f"✅ 模型已保存: {filepath}")
                        st.rerun()
            
            versions = load_model_versions()
            
            if versions:
                st.subheader("模型版本列表")
                for i, version in enumerate(versions):
                    with st.expander(f"版本 {i+1}: {version['timestamp']} ({version['model_type']})", expanded=(i==0)):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("训练集R²", version['train_metrics'].get('R²', 'N/A'))
                        with col2:
                            st.metric("测试集R²", version['test_metrics'].get('R²', 'N/A'))
                        with col3:
                            st.metric("测试集RMSE", version['test_metrics'].get('RMSE', 'N/A'))
                        
                        if st.button(f"加载此版本", key=f"load_{i}"):
                            st.session_state.selected_model = version['data']
                            st.success("✅ 模型已加载")
            else:
                st.info("暂无保存的模型版本。")
        
        with tab4:
            st.subheader("模型详情")
            
            if st.session_state.selected_model is not None:
                model_result = st.session_state.selected_model
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**训练集指标**")
                    for k, v in model_result['train_metrics'].items():
                        st.write(f"- {k}: {v}")
                with col2:
                    st.write("**测试集指标**")
                    for k, v in model_result['test_metrics'].items():
                        st.write(f"- {k}: {v}")
                
                st.subheader("预测值 vs 实际值 (测试集)")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=model_result['y_test'],
                    y=model_result['y_test_pred'],
                    mode='markers',
                    name='预测点',
                    marker=dict(color='blue', opacity=0.6),
                ))
                
                min_val = min(min(model_result['y_test']), min(model_result['y_test_pred']))
                max_val = max(max(model_result['y_test']), max(model_result['y_test_pred']))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='完美预测线',
                    line=dict(color='red', dash='dash'),
                ))
                
                fig.update_layout(
                    xaxis_title='实际投药量 (mg/L)',
                    yaxis_title='预测投药量 (mg/L)',
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("残差分布")
                
                residuals = model_result['residuals']
                fig = px.histogram(
                    x=residuals,
                    nbins=30,
                    title='残差分布直方图',
                    labels={'x': '残差 (mg/L)', 'y': '频次'},
                )
                fig.add_vline(x=0, line_dash='dash', line_color='red')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请先选择并加载一个模型。")
