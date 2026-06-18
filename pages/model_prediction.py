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
    get_feature_importance,
    find_similar_conditions,
    FEATURE_COLS_CN,
    TARGET_COL_CN,
    MODEL_TYPES,
)

if 'selected_versions' not in st.session_state:
    st.session_state.selected_versions = []


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
                
                st.divider()
                st.subheader("🎯 特征重要性分析")
                
                model_names_for_importance = list(st.session_state.model_results.keys())
                selected_model_for_importance = st.selectbox(
                    "选择模型查看特征重要性",
                    model_names_for_importance,
                    index=0,
                    key="importance_model_select"
                )
                
                if selected_model_for_importance in st.session_state.model_results:
                    model_result = st.session_state.model_results[selected_model_for_importance]
                    importance_data = get_feature_importance(model_result)
                    
                    if importance_data is not None:
                        if importance_data['type'] == 'importance':
                            df_imp = importance_data['data']
                            colors = px.colors.sequential.Reds[:len(df_imp)]
                            
                            fig_imp = go.Figure(go.Bar(
                                y=df_imp['feature'],
                                x=df_imp['importance'],
                                orientation='h',
                                marker=dict(
                                    color=colors,
                                    line=dict(color='rgba(0,0,0,0.3)', width=1),
                                ),
                                hovertemplate='<b>%{y}</b><br>重要性: %{x:.4f}<extra></extra>',
                            ))
                            fig_imp.update_layout(
                                title=f"{selected_model_for_importance} - 特征重要性排名",
                                xaxis_title="重要性",
                                yaxis_title="特征",
                                height=400,
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
                            
                        elif importance_data['type'] == 'coefficient':
                            df_coef = importance_data['data']
                            colors = ['#FF6B6B' if c < 0 else '#4ECDC4' for c in df_coef['coefficient']]
                            
                            fig_coef = go.Figure(go.Bar(
                                y=df_coef['feature'],
                                x=df_coef['abs_coefficient'],
                                orientation='h',
                                marker=dict(
                                    color=colors,
                                    line=dict(color='rgba(0,0,0,0.3)', width=1),
                                ),
                                customdata=df_coef['coefficient'],
                                hovertemplate='<b>%{y}</b><br>系数绝对值: %{x:.4f}<br>系数值: %{customdata:.4f}<extra></extra>',
                            ))
                            fig_coef.update_layout(
                                title=f"{selected_model_for_importance} - 回归系数绝对值排名<br><span style='font-size:12px;color:gray'>红色=负系数，青色=正系数</span>",
                                xaxis_title="系数绝对值",
                                yaxis_title="特征",
                                height=400,
                            )
                            st.plotly_chart(fig_coef, use_container_width=True)
                    else:
                        st.info("该模型类型不支持特征重要性分析。")
        
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
                    
                    st.divider()
                    st.subheader("📋 历史相似工况匹配")
                    
                    similar_records = find_similar_conditions(
                        df, features, model_result['feature_cols'], top_k=5
                    )
                    
                    if similar_records is not None and len(similar_records) > 0:
                        st.info(f"找到 {len(similar_records)} 条与当前工况最相似的历史记录（按欧氏距离从近到远排序）")
                        
                        display_cols = similar_records.columns.tolist()
                        rename_map = {c: c for c in display_cols}
                        rename_map['distance'] = '欧氏距离'
                        rename_map[TARGET_COL_CN] = '实际投药量 (mg/L)'
                        
                        display_df = similar_records.copy()
                        display_df = display_df.rename(columns=rename_map)
                        
                        for col in display_df.columns:
                            if col != '欧氏距离':
                                display_df[col] = display_df[col].round(4)
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                '欧氏距离': st.column_config.NumberColumn(
                                    '欧氏距离',
                                    format='%.2f',
                                ),
                            }
                        )
                        
                        actual_dosages = similar_records[TARGET_COL_CN].values
                        st.caption(f"历史相似工况实际投药量范围: {actual_dosages.min():.2f} ~ {actual_dosages.max():.2f} mg/L，均值: {actual_dosages.mean():.2f} mg/L")
                    else:
                        st.warning("未找到足够的历史数据进行相似工况匹配。")

                    if 'network_hydraulic_summary' in st.session_state and st.session_state.network_hydraulic_summary is not None:
                        st.divider()
                        st.subheader("📢 管网水力风险提示")

                        summary = st.session_state.network_hydraulic_summary
                        min_vel_pipe = summary.get('min_velocity_pipe', {})
                        min_press_node = summary.get('min_pressure_node', {})

                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.info(f"""
                            **最低流速管段**
                            - 管段编号: {min_vel_pipe.get('pipe_id', 'N/A')}
                            - 流速值: {min_vel_pipe.get('velocity', 0):.4f} m/s
                            - 停留时间: {min_vel_pipe.get('travel_time_h', 0):.4f} 小时
                            """)
                        with col_r2:
                            st.info(f"""
                            **最低压力节点**
                            - 节点名称: {min_press_node.get('node_name', 'N/A')}
                            - 压力值: {min_press_node.get('pressure', 0):.2f} m
                            """)

                        vel_warning = min_vel_pipe.get('velocity', 0) < 0.3
                        press_warning = min_press_node.get('pressure', 0) < 10
                        if vel_warning or press_warning:
                            warning_msg = []
                            if vel_warning:
                                warning_msg.append("最低流速低于0.3m/s，存在滞流风险")
                            if press_warning:
                                warning_msg.append("最低压力低于10m，存在供水不足风险")
                            st.warning("⚠️ " + "；".join(warning_msg))
            else:
                st.info("请先在'模型训练与对比'标签页中训练模型。")
        
        with tab3:
            st.subheader("模型版本管理")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("保留版本数", "最近3个版本")
            with col2:
                if st.button("💾 保存当前模型", disabled=st.session_state.selected_model is None):
                    if st.session_state.selected_model is not None:
                        filepath, timestamp = save_model(st.session_state.selected_model)
                        st.success(f"✅ 模型已保存: {filepath}")
                        st.rerun()
            with col3:
                selected_count = len(st.session_state.selected_versions)
                compare_disabled = selected_count != 2
                compare_help = "请恰好选择两个版本进行对比" if selected_count != 2 else "点击对比选中的两个版本"
                if st.button("📊 对比选中版本", disabled=compare_disabled, help=compare_help, type="primary"):
                    pass
            
            versions = load_model_versions()
            
            if versions:
                st.subheader("模型版本列表")
                
                if 'selected_versions' not in st.session_state:
                    st.session_state.selected_versions = []
                
                for i, version in enumerate(versions):
                    version_key = f"{version['filename']}_{i}"
                    is_checked = version_key in st.session_state.selected_versions
                    
                    with st.expander(f"版本 {i+1}: {version['timestamp']} ({version['model_type']})", expanded=(i==0)):
                        col1, col2, col3, col4 = st.columns([3, 3, 3, 1])
                        with col1:
                            st.metric("训练集R²", version['train_metrics'].get('R²', 'N/A'))
                        with col2:
                            st.metric("测试集R²", version['test_metrics'].get('R²', 'N/A'))
                        with col3:
                            st.metric("测试集RMSE", version['test_metrics'].get('RMSE', 'N/A'))
                        with col4:
                            select = st.checkbox("选择", value=is_checked, key=f"sel_{version_key}")
                            if select and version_key not in st.session_state.selected_versions:
                                st.session_state.selected_versions.append(version_key)
                            elif not select and version_key in st.session_state.selected_versions:
                                st.session_state.selected_versions.remove(version_key)
                        
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button(f"加载此版本", key=f"load_{i}"):
                                st.session_state.selected_model = version['data']
                                st.success("✅ 模型已加载")
                        with col_btn2:
                            if st.button(f"删除此版本", key=f"del_{i}"):
                                try:
                                    os.remove(version['filepath'])
                                    st.success(f"✅ 版本已删除: {version['filename']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
                
                selected_count = len(st.session_state.selected_versions)
                if selected_count == 0:
                    st.info("💡 请勾选需要对比的版本（恰好两个）")
                elif selected_count == 1:
                    st.info(f"💡 已选择 1 个版本，请再选择 1 个进行对比")
                elif selected_count == 2:
                    st.success(f"✅ 已选择 2 个版本，点击上方'对比选中版本'按钮进行对比")
                    
                    if st.button("📊 生成对比图表", type="primary", use_container_width=True):
                        version1 = None
                        version2 = None
                        for i, v in enumerate(versions):
                            vk = f"{v['filename']}_{i}"
                            if vk == st.session_state.selected_versions[0]:
                                version1 = v
                            elif vk == st.session_state.selected_versions[1]:
                                version2 = v
                        
                        if version1 and version2:
                            col_scatter, col_radar = st.columns(2)
                            
                            with col_scatter:
                                st.subheader("预测值 vs 真实值 (测试集)")
                                fig_scatter = go.Figure()
                                
                                y_test_1 = version1['y_test']
                                y_pred_1 = version1['y_test_pred']
                                y_test_2 = version2['y_test']
                                y_pred_2 = version2['y_test_pred']
                                
                                fig_scatter.add_trace(go.Scatter(
                                    x=y_test_1,
                                    y=y_pred_1,
                                    mode='markers',
                                    name=f"版本1: {version1['timestamp'][:12]}",
                                    marker=dict(color='rgba(0, 100, 255, 0.6)', size=8),
                                ))
                                
                                fig_scatter.add_trace(go.Scatter(
                                    x=y_test_2,
                                    y=y_pred_2,
                                    mode='markers',
                                    name=f"版本2: {version2['timestamp'][:12]}",
                                    marker=dict(color='rgba(255, 100, 0, 0.6)', size=8),
                                ))
                                
                                all_vals = y_test_1 + y_pred_1 + y_test_2 + y_pred_2
                                min_val = min(all_vals)
                                max_val = max(all_vals)
                                fig_scatter.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='完美预测线',
                                    line=dict(color='red', dash='dash'),
                                ))
                                
                                fig_scatter.update_layout(
                                    xaxis_title='实际投药量 (mg/L)',
                                    yaxis_title='预测投药量 (mg/L)',
                                    height=500,
                                    legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            with col_radar:
                                st.subheader("性能指标雷达图对比")
                                metrics = ['R²', 'RMSE', 'MAE']
                                
                                v1_test = version1['test_metrics']
                                v2_test = version2['test_metrics']
                                
                                v1_values = [
                                    v1_test.get('R²', 0),
                                    1 / (1 + v1_test.get('RMSE', 1)),
                                    1 / (1 + v1_test.get('MAE', 1)),
                                ]
                                v2_values = [
                                    v2_test.get('R²', 0),
                                    1 / (1 + v2_test.get('RMSE', 1)),
                                    1 / (1 + v2_test.get('MAE', 1)),
                                ]
                                
                                fig_radar = go.Figure()
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=v1_values,
                                    theta=metrics,
                                    fill='toself',
                                    name=f"版本1: {version1['timestamp'][:12]}",
                                    fillcolor='rgba(0, 100, 255, 0.3)',
                                    line=dict(color='rgba(0, 100, 255, 1)'),
                                ))
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=v2_values,
                                    theta=metrics,
                                    fill='toself',
                                    name=f"版本2: {version2['timestamp'][:12]}",
                                    fillcolor='rgba(255, 100, 0, 0.3)',
                                    line=dict(color='rgba(255, 100, 0, 1)'),
                                ))
                                
                                fig_radar.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1],
                                        ),
                                    ),
                                    height=500,
                                    legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)
                                
                                st.divider()
                                st.subheader("📊 指标数值对比")
                                compare_table = pd.DataFrame({
                                    '指标': ['R²', 'RMSE', 'MAE'],
                                    f"版本1 ({version1['timestamp'][:12]})": [
                                        v1_test.get('R²', 'N/A'),
                                        v1_test.get('RMSE', 'N/A'),
                                        v1_test.get('MAE', 'N/A'),
                                    ],
                                    f"版本2 ({version2['timestamp'][:12]})": [
                                        v2_test.get('R²', 'N/A'),
                                        v2_test.get('RMSE', 'N/A'),
                                        v2_test.get('MAE', 'N/A'),
                                    ],
                                })
                                st.dataframe(compare_table, use_container_width=True, hide_index=True)
                                
                                st.caption("说明：R²越大越好，RMSE和MAE越小越好。雷达图中为便于可视化，RMSE和MAE已转换为1/(1+值)形式。")
                else:
                    st.warning(f"⚠️ 已选择 {selected_count} 个版本，请恰好选择 2 个版本进行对比")
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
