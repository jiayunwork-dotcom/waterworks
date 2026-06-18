import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import (
    load_data,
    process_data,
    get_statistics,
    get_missing_color,
    get_latest_values,
    PROCESS_STAGES,
    REVERSE_MAPPING,
    detect_trend_warnings,
)


st.set_page_config(
    page_title="自来水厂工艺流程监控系统",
    page_icon="💧",
    layout="wide",
)


st.title("💧 自来水厂工艺流程监控与混凝剂投药优化系统")

st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择功能模块",
    [
        "数据导入与工艺概览",
        "投药量预测模型",
        "余氯衰减与CT值计算",
        "水质达标监控",
        "工艺参数关联分析",
        "能耗分析",
        "历史趋势与季节性分析",
        "报告导出",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("数据上传")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'energy_df' not in st.session_state:
    st.session_state.energy_df = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

uploaded_file = st.sidebar.file_uploader(
    "上传水质监测数据 (CSV)",
    type=['csv'],
    help="包含时间戳和各工艺段监测指标的CSV文件",
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        df = process_data(df)
        st.session_state.df = df
        st.sidebar.success(f"✅ 数据加载成功，共 {len(df)} 条记录")

if st.sidebar.button("📥 加载示例数据", use_container_width=True):
    try:
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data.csv')
        df = pd.read_csv(sample_path)
        df = process_data(df)
        st.session_state.df = df
        st.sidebar.success(f"✅ 示例数据加载成功，共 {len(df)} 条记录")
    except Exception as e:
        st.sidebar.error(f"加载失败: {e}")

energy_file = st.sidebar.file_uploader(
    "上传能耗数据 (CSV)",
    type=['csv'],
    help="包含各工艺环节电耗数据的CSV文件",
)

if energy_file is not None:
    from utils.energy_utils import load_energy_data
    energy_df = load_energy_data(energy_file)
    if energy_df is not None:
        st.session_state.energy_df = energy_df
        st.sidebar.success(f"✅ 能耗数据加载成功，共 {len(energy_df)} 条记录")

if st.session_state.df is not None:
    df = st.session_state.df
    trend_warnings = detect_trend_warnings(df)
    if trend_warnings:
        with st.container():
            st.warning("⚠️ 趋势预警")
            for w in trend_warnings:
                st.write(f"- **{w['指标']}** 呈{w['趋势']}趋势，当前值: {w['当前值']}，距离限值: {w['距离限值']}")
        st.divider()


if page == "数据导入与工艺概览":
    st.header("📊 数据导入与工艺段配置")
    
    if st.session_state.df is None:
        st.info("👈 请在左侧上传CSV数据文件，或点击'加载示例数据'按钮开始使用。")
        
        with st.expander("📋 数据格式说明", expanded=True):
            st.markdown("""
            CSV文件需包含以下列（至少包含时间戳列和主要监测指标）：
            
            | 列名 | 说明 | 单位 |
            |------|------|------|
            | 时间 | 时间戳 | yyyy-MM-dd HH:mm:ss |
            | 源水浊度 | 原水浊度 | NTU |
            | 源水pH | 原水pH值 | - |
            | 源水温度 | 原水温度 | ℃ |
            | 源水氨氮 | 原水氨氮含量 | mg/L |
            | 源水COD | 原水COD | mg/L |
            | 源水有机物 | UV254 | - |
            | 混凝剂投加量 | 混凝剂投加量 | mg/L |
            | 沉后浊度 | 沉淀后浊度 | NTU |
            | 滤后浊度 | 过滤后浊度 | NTU |
            | 出水浊度 | 出厂水浊度 | NTU |
            | 出水余氯 | 出厂水余氯 | mg/L |
            | 出水pH | 出厂水pH | - |
            """)
    else:
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["📈 数据概览", "📊 统计摘要", "🏭 工艺流程拓扑"])
        
        with tab1:
            st.subheader("数据预览")
            st.dataframe(df.head(20), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据条数", len(df))
            with col2:
                st.metric("指标列数", len(df.columns) - 1)
            with col3:
                time_range = f"{df['时间'].min().strftime('%Y-%m-%d')} ~ {df['时间'].max().strftime('%Y-%m-%d')}"
                st.metric("时间范围", time_range)
        
        with tab2:
            st.subheader("各列统计摘要")
            
            stats_df = get_statistics(df)
            
            def highlight_missing(row):
                missing_rate = row['缺失率(%)']
                if missing_rate >= 30:
                    return ['background-color: #ffcccc'] * len(row)
                elif missing_rate >= 10:
                    return ['background-color: #fff2cc'] * len(row)
                return [''] * len(row)
            
            styled_stats = stats_df.style.apply(highlight_missing, axis=1)
            st.dataframe(styled_stats, use_container_width=True, hide_index=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ 正常列 (缺失率<10%): {len(stats_df[stats_df['状态']=='正常'])} 列")
            with col2:
                st.warning(f"⚠️ 警告列 (缺失率10-30%): {len(stats_df[stats_df['状态']=='警告'])} 列")
            with col3:
                st.error(f"❌ 严重列 (缺失率>30%): {len(stats_df[stats_df['状态']=='严重'])} 列")
            
            st.info("💡 建议：缺失率超过30%的列不建议参与建模计算。")
        
        with tab3:
            st.subheader("工艺流程拓扑图")
            
            latest = get_latest_values(df)
            
            stages_info = []
            for stage in PROCESS_STAGES:
                stage_metrics = []
                for metric_key in stage['metrics']:
                    cn_name = REVERSE_MAPPING.get(metric_key, metric_key)
                    if cn_name in latest:
                        val = latest[cn_name]
                        if pd.notna(val):
                            if isinstance(val, (int, float)):
                                stage_metrics.append(f"{cn_name}: {val:.3f}")
                            else:
                                stage_metrics.append(f"{cn_name}: {val}")
                stages_info.append({
                    'name': stage['name'],
                    'metrics': stage_metrics,
                })
            
            fig = go.Figure()
            
            x_positions = list(range(len(stages_info)))
            y_base = 0
            
            for i, stage in enumerate(stages_info):
                x = x_positions[i]
                
                metric_text = "<br>".join(stage['metrics']) if stage['metrics'] else "—"
                
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y_base],
                    mode='markers+text',
                    marker=dict(
                        size=50,
                        color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                        line=dict(color='white', width=2),
                    ),
                    text=[f"<b>{stage['name']}</b><br>{metric_text}"],
                    textposition='bottom center',
                    hoverinfo='text',
                    showlegend=False,
                ))
                
                if i < len(stages_info) - 1:
                    fig.add_annotation(
                        x=x_positions[i+1] - 0.25,
                        y=y_base,
                        ax=x + 0.25,
                        ay=y_base,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=2,
                        arrowcolor='gray',
                        showarrow=True,
                    )
            
            fig.update_layout(
                title="工艺流程拓扑",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-0.5, len(stages_info) - 0.5],
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-1, 1],
                ),
                height=400,
                margin=dict(l=20, r=20, t=50, b=100),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("图中显示各工艺段最新监测数据，节点从左到右依次为：取水→混凝→沉淀→过滤→消毒→清水池→供水")

elif page == "投药量预测模型":
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages', 'model_prediction.py'), encoding='utf-8').read())

elif page == "余氯衰减与CT值计算":
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages', 'chlorine_ct.py'), encoding='utf-8').read())

elif page == "水质达标监控":
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages', 'water_quality.py'), encoding='utf-8').read())

elif page == "工艺参数关联分析":
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages', 'correlation_analysis.py'), encoding='utf-8').read())

elif page == "能耗分析":
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages', 'energy_analysis.py'), encoding='utf-8').read())

elif page == "历史趋势与季节性分析":
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages', 'trend_seasonality.py'), encoding='utf-8').read())

elif page == "报告导出":
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages', 'report_export.py'), encoding='utf-8').read())
