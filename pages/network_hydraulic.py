import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import json
import io
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network_utils import (
    validate_network,
    hardy_cross_calculation,
    calculate_water_quality,
    sensitivity_analysis,
    compute_network_layout,
    genetic_algorithm_optimization,
    calculate_total_cost,
    calculate_pipe_cost,
    CANDIDATE_DIAMETERS,
)


st.header("🔧 管网水力模型仿真")

if 'network_nodes' not in st.session_state:
    st.session_state.network_nodes = pd.DataFrame(
        columns=['节点名称', '类型', '用水量(m³/h)', '水头(m)', '地面标高(m)']
    )
if 'network_pipes' not in st.session_state:
    st.session_state.network_pipes = pd.DataFrame(
        columns=['起始节点', '终止节点', '管径(mm)', '管长(m)', '粗糙系数']
    )
if 'hydraulic_results' not in st.session_state:
    st.session_state.hydraulic_results = None
if 'water_quality_results' not in st.session_state:
    st.session_state.water_quality_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📐 管网拓扑定义", "💧 水力计算", "🧪 水质衰减耦合", "📊 敏感性分析", "⚙️ 管径优化"])

with tab1:
    st.subheader("节点配置")

    node_col1, node_col2 = st.columns([3, 1])
    with node_col2:
        st.markdown("##### 添加节点")
        new_node_name = st.text_input("节点名称", key="new_node_name")
        new_node_type = st.selectbox("类型", ["需求节点", "水源节点"], key="new_node_type")
        new_node_demand = st.number_input("用水量 (m³/h)", min_value=0.0, value=10.0, step=1.0, key="new_node_demand")
        new_node_head = st.number_input("水头 (m)", min_value=0.0, value=30.0, step=1.0, key="new_node_head",
                                        help="仅水源节点需要设置水头值")
        new_node_elev = st.number_input("地面标高 (m)", min_value=0.0, value=0.0, step=0.5, key="new_node_elev")

        if st.button("➕ 添加节点", use_container_width=True):
            if new_node_name:
                existing = st.session_state.network_nodes
                if new_node_name in existing['节点名称'].values:
                    st.warning(f"节点 '{new_node_name}' 已存在")
                else:
                    type_val = "水源" if new_node_type == "水源节点" else "需求"
                    new_row = pd.DataFrame([{
                        '节点名称': new_node_name,
                        '类型': type_val,
                        '用水量(m³/h)': new_node_demand if type_val == "需求" else 0.0,
                        '水头(m)': new_node_head if type_val == "水源" else 0.0,
                        '地面标高(m)': new_node_elev,
                    }])
                    st.session_state.network_nodes = pd.concat(
                        [existing, new_row], ignore_index=True
                    )
                    st.rerun()
            else:
                st.warning("请输入节点名称")

    with node_col1:
        if len(st.session_state.network_nodes) > 0:
            edited_nodes = st.data_editor(
                st.session_state.network_nodes,
                use_container_width=True,
                num_rows="dynamic",
                key="nodes_editor",
            )
            st.session_state.network_nodes = edited_nodes
        else:
            st.info("暂无节点，请在右侧添加")

    st.divider()
    st.subheader("管段配置")

    pipe_col1, pipe_col2 = st.columns([3, 1])
    with pipe_col2:
        st.markdown("##### 添加管段")
        node_names = st.session_state.network_nodes['节点名称'].tolist()
        if len(node_names) >= 2:
            new_pipe_start = st.selectbox("起始节点", node_names, key="new_pipe_start")
            new_pipe_end = st.selectbox("终止节点", node_names, key="new_pipe_end")
            new_pipe_diam = st.number_input("管径 (mm)", min_value=50, value=300, step=50, key="new_pipe_diam")
            new_pipe_len = st.number_input("管长 (m)", min_value=1.0, value=500.0, step=10.0, key="new_pipe_len")
            new_pipe_c = st.number_input("粗糙系数 (H-W)", min_value=50, value=130, step=5, key="new_pipe_c")

            if st.button("➕ 添加管段", use_container_width=True):
                if new_pipe_start == new_pipe_end:
                    st.warning("起始节点和终止节点不能相同")
                else:
                    existing = st.session_state.network_pipes
                    dup = existing[
                        ((existing['起始节点'] == new_pipe_start) & (existing['终止节点'] == new_pipe_end)) |
                        ((existing['起始节点'] == new_pipe_end) & (existing['终止节点'] == new_pipe_start))
                    ]
                    if not dup.empty:
                        st.warning(f'节点 {new_pipe_start} 与 {new_pipe_end} 之间已存在管段，不可重复添加')
                    else:
                        new_row = pd.DataFrame([{
                            '起始节点': new_pipe_start,
                            '终止节点': new_pipe_end,
                            '管径(mm)': new_pipe_diam,
                            '管长(m)': new_pipe_len,
                            '粗糙系数': new_pipe_c,
                        }])
                        st.session_state.network_pipes = pd.concat(
                            [existing, new_row], ignore_index=True
                        )
                        st.rerun()
        else:
            st.info("请先添加至少2个节点")

    with pipe_col1:
        if len(st.session_state.network_pipes) > 0:
            edited_pipes = st.data_editor(
                st.session_state.network_pipes,
                use_container_width=True,
                num_rows="dynamic",
                key="pipes_editor",
            )
            st.session_state.network_pipes = edited_pipes
        else:
            st.info("暂无管段，请在右侧添加")

    st.divider()

    col_load, col_clear = st.columns(2)
    with col_load:
        if st.button("📋 加载示例管网", use_container_width=True):
            st.session_state.network_nodes = pd.DataFrame([
                {'节点名称': 'S1', '类型': '水源', '用水量(m³/h)': 0.0, '水头(m)': 40.0, '地面标高(m)': 5.0},
                {'节点名称': 'N1', '类型': '需求', '用水量(m³/h)': 20.0, '水头(m)': 0.0, '地面标高(m)': 6.0},
                {'节点名称': 'N2', '类型': '需求', '用水量(m³/h)': 30.0, '水头(m)': 0.0, '地面标高(m)': 7.0},
                {'节点名称': 'N3', '类型': '需求', '用水量(m³/h)': 15.0, '水头(m)': 0.0, '地面标高(m)': 8.0},
                {'节点名称': 'N4', '类型': '需求', '用水量(m³/h)': 25.0, '水头(m)': 0.0, '地面标高(m)': 6.5},
                {'节点名称': 'N5', '类型': '需求', '用水量(m³/h)': 10.0, '水头(m)': 0.0, '地面标高(m)': 9.0},
            ])
            st.session_state.network_pipes = pd.DataFrame([
                {'起始节点': 'S1', '终止节点': 'N1', '管径(mm)': 400, '管长(m)': 300, '粗糙系数': 130},
                {'起始节点': 'S1', '终止节点': 'N2', '管径(mm)': 350, '管长(m)': 400, '粗糙系数': 130},
                {'起始节点': 'N1', '终止节点': 'N3', '管径(mm)': 250, '管长(m)': 500, '粗糙系数': 130},
                {'起始节点': 'N2', '终止节点': 'N4', '管径(mm)': 250, '管长(m)': 450, '粗糙系数': 130},
                {'起始节点': 'N3', '终止节点': 'N5', '管径(mm)': 200, '管长(m)': 350, '粗糙系数': 130},
                {'起始节点': 'N4', '终止节点': 'N5', '管径(mm)': 200, '管长(m)': 300, '粗糙系数': 130},
                {'起始节点': 'N1', '终止节点': 'N4', '管径(mm)': 200, '管长(m)': 600, '粗糙系数': 130},
            ])
            st.session_state.hydraulic_results = None
            st.session_state.water_quality_results = None
            st.rerun()
    with col_clear:
        if st.button("🗑️ 清空管网", use_container_width=True):
            st.session_state.network_nodes = pd.DataFrame(
                columns=['节点名称', '类型', '用水量(m³/h)', '水头(m)', '地面标高(m)']
            )
            st.session_state.network_pipes = pd.DataFrame(
                columns=['起始节点', '终止节点', '管径(mm)', '管长(m)', '粗糙系数']
            )
            st.session_state.hydraulic_results = None
            st.session_state.water_quality_results = None
            st.rerun()

    st.divider()
    st.subheader("管网拓扑图")

    nodes_df = st.session_state.network_nodes
    pipes_df = st.session_state.network_pipes

    if len(nodes_df) > 0 and len(pipes_df) > 0:
        errors = validate_network(nodes_df, pipes_df)
        if errors:
            for e in errors:
                st.error(e)
        else:
            pos = compute_network_layout(nodes_df, pipes_df)

            fig = go.Figure()

            for _, row in pipes_df.iterrows():
                start = row['起始节点']
                end = row['终止节点']
                if start in pos and end in pos:
                    x0, y0 = pos[start]
                    x1, y1 = pos[end]
                    fig.add_trace(go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines+text',
                        line=dict(color='gray', width=3),
                        text=[None, f"DN{int(row['管径(mm)'])}"],
                        textposition='top center',
                        textfont=dict(size=10, color='gray'),
                        hoverinfo='text',
                        hovertext=f"{start}→{end}<br>管径: {int(row['管径(mm)'])}mm<br>管长: {row['管长(m)']}m<br>H-W系数: {row['粗糙系数']}",
                        showlegend=False,
                    ))

            for _, row in nodes_df.iterrows():
                name = row['节点名称']
                if name in pos:
                    x, y = pos[name]
                    is_source = row['类型'] == '水源'
                    color = 'blue' if is_source else 'green'
                    size = 16

                    label = f"{name}"
                    if is_source:
                        label += f"<br>水源 H={row['水头(m)']}m"
                    else:
                        label += f"<br>需求 {row['用水量(m³/h)']}m³/h"

                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(size=size, color=color,
                                    line=dict(color='white', width=2)),
                        text=[label],
                        textposition='bottom center',
                        textfont=dict(size=10),
                        hoverinfo='text',
                        showlegend=False,
                    ))

            fig.update_layout(
                title="管网拓扑图",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           scaleanchor='x', scaleratio=1),
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("请添加节点和管段后查看管网拓扑图")

    st.divider()
    st.subheader("📤 管网拓扑导入导出")

    col_export, col_import = st.columns(2)

    with col_export:
        st.markdown("##### 导出当前管网")
        if len(st.session_state.network_nodes) > 0 or len(st.session_state.network_pipes) > 0:
            export_data = {
                'nodes': st.session_state.network_nodes.to_dict('records'),
                'pipes': st.session_state.network_pipes.to_dict('records'),
            }
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="💾 导出为JSON文件",
                data=json_str,
                file_name="network_topology.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("暂无管网数据可导出")

    with col_import:
        st.markdown("##### 导入管网")
        uploaded_file = st.file_uploader("选择JSON文件", type=['json'], key="network_import")
        if uploaded_file is not None:
            try:
                bytes_data = uploaded_file.read()
                data = json.load(io.BytesIO(bytes_data))

                required_node_fields = ['节点名称', '类型', '用水量(m³/h)', '水头(m)', '地面标高(m)']
                required_pipe_fields = ['起始节点', '终止节点', '管径(mm)', '管长(m)', '粗糙系数']

                errors = []
                if 'nodes' not in data:
                    errors.append("JSON文件缺少 'nodes' 字段")
                else:
                    if not isinstance(data['nodes'], list):
                        errors.append("'nodes' 字段必须是数组类型")
                    else:
                        for i, node in enumerate(data['nodes']):
                            if not isinstance(node, dict):
                                errors.append(f"nodes[{i}] 必须是对象类型")
                            else:
                                for field in required_node_fields:
                                    if field not in node:
                                        errors.append(f"nodes[{i}] 缺少必要字段: '{field}'")

                if 'pipes' not in data:
                    errors.append("JSON文件缺少 'pipes' 字段")
                else:
                    if not isinstance(data['pipes'], list):
                        errors.append("'pipes' 字段必须是数组类型")
                    else:
                        for i, pipe in enumerate(data['pipes']):
                            if not isinstance(pipe, dict):
                                errors.append(f"pipes[{i}] 必须是对象类型")
                            else:
                                for field in required_pipe_fields:
                                    if field not in pipe:
                                        errors.append(f"pipes[{i}] 缺少必要字段: '{field}'")

                if errors:
                    st.error("❌ 导入失败，存在以下错误：")
                    for err in errors:
                        st.error(f"  - {err}")
                else:
                    if st.button("✅ 确认导入并替换当前管网", use_container_width=True, type="primary"):
                        new_nodes = pd.DataFrame(data['nodes'])
                        new_pipes = pd.DataFrame(data['pipes'])

                        validate_errors = validate_network(new_nodes, new_pipes)
                        if validate_errors:
                            st.error("❌ 管网拓扑校验失败：")
                            for err in validate_errors:
                                st.error(f"  - {err}")
                        else:
                            st.session_state.network_nodes = new_nodes
                            st.session_state.network_pipes = new_pipes
                            st.session_state.hydraulic_results = None
                            st.session_state.water_quality_results = None
                            st.success(f"✅ 导入成功！共 {len(new_nodes)} 个节点，{len(new_pipes)} 条管段")
                            st.rerun()

            except json.JSONDecodeError as e:
                st.error(f"❌ JSON格式解析失败: {str(e)}")
            except Exception as e:
                st.error(f"❌ 导入失败: {str(e)}")

with tab2:
    st.subheader("稳态水力计算 (Hardy-Cross法)")

    nodes_df = st.session_state.network_nodes
    pipes_df = st.session_state.network_pipes

    if len(nodes_df) == 0 or len(pipes_df) == 0:
        st.info('请先在"管网拓扑定义"页签中配置管网结构')
    else:
        errors = validate_network(nodes_df, pipes_df)
        if errors:
            for e in errors:
                st.error(e)
            st.warning("请修正以上错误后再进行水力计算")
        else:
            st.markdown("""
            **Hazen-Williams公式：**
            
            $h_f = \\frac{10.67 \\cdot L \\cdot Q^{1.852}}{C^{1.852} \\cdot D^{4.87}}$
            
            其中：L=管长(m)，Q=流量(m³/s)，C=Hazen-Williams粗糙系数，D=管径(m)
            """)

            col_conv1, col_conv2 = st.columns(2)
            with col_conv1:
                tol = st.number_input("收敛阈值 (m³/s)", min_value=0.0001, value=0.001, step=0.0001,
                                       format="%.4f")
            with col_conv2:
                max_iter = st.number_input("最大迭代次数", min_value=10, value=500, step=10)

            if st.button("🔄 执行水力计算", type="primary", use_container_width=True):
                with st.spinner("正在执行Hardy-Cross迭代计算..."):
                    result = hardy_cross_calculation(nodes_df, pipes_df, max_iter=max_iter, tol=tol)
                    st.session_state.hydraulic_results = result
                    st.session_state.water_quality_results = None

                    pipe_results_df = pd.DataFrame(result['pipe_results'])
                    if not pipe_results_df.empty:
                        pipe_results_df['流速状态'] = pipe_results_df['流速(m/s)'].apply(
                            lambda v: '滞流风险' if v < 0.3 else ('冲刷风险' if v > 2.5 else '正常')
                        )
                        min_velocity_row = pipe_results_df.loc[pipe_results_df['流速(m/s)'].idxmin()]
                        pipe_id = f"{min_velocity_row['起始节点']}→{min_velocity_row['终止节点']}"
                        min_velocity = min_velocity_row['流速(m/s)']
                        pipe_len = min_velocity_row['管长(m)']
                        travel_time_h = (pipe_len / (min_velocity * 3600)) if min_velocity > 0 else 9999.0

                        node_results_df = pd.DataFrame(result['node_results'])
                        demand_nodes = node_results_df[node_results_df['类型'] == '需求']
                        if not demand_nodes.empty:
                            min_pressure_row = demand_nodes.loc[demand_nodes['压力(m)'].idxmin()]
                            min_pressure_node = min_pressure_row['节点名称']
                            min_pressure = min_pressure_row['压力(m)']
                        else:
                            min_pressure_node = 'N/A'
                            min_pressure = 0.0

                        st.session_state.network_hydraulic_summary = {
                            'min_velocity_pipe': {
                                'pipe_id': pipe_id,
                                'velocity': float(min_velocity),
                                'travel_time_h': float(travel_time_h),
                            },
                            'min_pressure_node': {
                                'node_name': min_pressure_node,
                                'pressure': float(min_pressure),
                            }
                        }

            result = st.session_state.hydraulic_results
            if result is not None:
                if result['converged']:
                    st.success(f"✅ 计算收敛，共迭代 {result['iterations']} 次")
                else:
                    st.warning(f"⚠️ 迭代未收敛（已迭代 {result['iterations']} 次），结果可能不准确，请检查管网参数")

                st.divider()
                st.subheader("管段计算结果")

                pipe_df = pd.DataFrame(result['pipe_results'])
                if not pipe_df.empty:
                    pipe_df['流速状态'] = pipe_df['流速(m/s)'].apply(
                        lambda v: '滞流风险' if v < 0.3 else ('冲刷风险' if v > 2.5 else '正常')
                    )
                    display_pipe = pipe_df[['起始节点', '终止节点', '管径(mm)', '管长(m)',
                                             '流量(L/s)', '流速(m/s)', '流速状态', '水头损失(m)']].copy()
                    display_pipe['流量(L/s)'] = display_pipe['流量(L/s)'].round(3)
                    display_pipe['流速(m/s)'] = display_pipe['流速(m/s)'].round(4)
                    display_pipe['水头损失(m)'] = display_pipe['水头损失(m)'].round(4)

                    def highlight_velocity_status(val):
                        if val == '滞流风险':
                            return 'background-color: #fff3cd; color: #856404'
                        elif val == '冲刷风险':
                            return 'background-color: #f8d7da; color: #721c24'
                        else:
                            return 'background-color: #d4edda; color: #155724'

                    styled_pipe = display_pipe.style.applymap(
                        highlight_velocity_status, subset=['流速状态']
                    )
                    st.dataframe(styled_pipe, use_container_width=True, hide_index=True)

                    stagnant = pipe_df[pipe_df['流速状态'] == '滞流风险']
                    scouring = pipe_df[pipe_df['流速状态'] == '冲刷风险']
                    if not stagnant.empty:
                        stagnant_pipes = [f"{r['起始节点']}→{r['终止节点']}" for _, r in stagnant.iterrows()]
                        st.warning(f"⚠️ 滞流风险管段 ({len(stagnant)}条，流速<0.3m/s): {', '.join(stagnant_pipes)}")
                    if not scouring.empty:
                        scouring_pipes = [f"{r['起始节点']}→{r['终止节点']}" for _, r in scouring.iterrows()]
                        st.error(f"🚨 冲刷风险管段 ({len(scouring)}条，流速>2.5m/s): {', '.join(scouring_pipes)}")

                st.divider()
                st.subheader("节点计算结果")

                node_df = pd.DataFrame(result['node_results'])
                if not node_df.empty:
                    display_node = node_df[['节点名称', '类型', '水头(m)', '压力(m)', '用水量(m³/h)']].copy()
                    display_node['水头(m)'] = display_node['水头(m)'].round(3)
                    display_node['压力(m)'] = display_node['压力(m)'].round(3)
                    st.dataframe(display_node, use_container_width=True, hide_index=True)

                    low_pressure = node_df[node_df['压力(m)'] < 10]
                    if not low_pressure.empty:
                        st.warning(f"⚠️ 以下节点压力低于10m: {', '.join(low_pressure['节点名称'].tolist())}")

                st.divider()
                st.subheader("管网压力热力图")

                pos = compute_network_layout(nodes_df, pipes_df)
                node_df_res = pd.DataFrame(result['node_results'])

                fig = go.Figure()

                pressures = node_df_res.set_index('节点名称')['压力(m)'].to_dict()
                p_min = min(pressures.values()) if pressures else 0
                p_max = max(pressures.values()) if pressures else 1

                diameters = pipes_df['管径(mm)'].values
                d_min = min(diameters)
                d_max = max(diameters)

                pipe_velocity_map = {}
                for pr in result['pipe_results']:
                    key = f"{pr['起始节点']}→{pr['终止节点']}"
                    pipe_velocity_map[key] = pr['流速(m/s)']

                legend_added = {'normal': False, 'stagnant': False, 'scouring': False}

                for _, row in pipes_df.iterrows():
                    start = row['起始节点']
                    end = row['终止节点']
                    if start in pos and end in pos:
                        x0, y0 = pos[start]
                        x1, y1 = pos[end]
                        d = row['管径(mm)']
                        if d_max > d_min:
                            line_width = 1.5 + 5 * (d - d_min) / (d_max - d_min)
                        else:
                            line_width = 3

                        key_fwd = f"{start}→{end}"
                        key_rev = f"{end}→{start}"
                        velocity = pipe_velocity_map.get(key_fwd, pipe_velocity_map.get(key_rev, 0))

                        if velocity < 0.3:
                            line_color = '#FFC107'
                            line_dash = 'dash'
                            status_text = '⚠️ 滞流风险'
                            legend_name = '滞流风险 (<0.3m/s)'
                            legend_key = 'stagnant'
                        elif velocity > 2.5:
                            line_color = '#DC3545'
                            line_dash = 'dash'
                            status_text = '🚨 冲刷风险'
                            legend_name = '冲刷风险 (>2.5m/s)'
                            legend_key = 'scouring'
                        else:
                            line_color = '#888888'
                            line_dash = 'solid'
                            status_text = '正常'
                            legend_name = '正常管段'
                            legend_key = 'normal'

                        show_legend = not legend_added[legend_key]
                        if show_legend:
                            legend_added[legend_key] = True

                        fig.add_trace(go.Scatter(
                            x=[x0, x1],
                            y=[y0, y1],
                            mode='lines',
                            line=dict(color=line_color, width=line_width, dash=line_dash),
                            hoverinfo='text',
                            hovertext=f"{start} → {end}<br>管径: {d} mm<br>流速: {velocity:.4f} m/s<br>状态: {status_text}",
                            showlegend=show_legend,
                            name=legend_name,
                        ))

                pressure_values = []
                node_x = []
                node_y = []
                node_text = []
                node_hover = []
                for _, row in node_df_res.iterrows():
                    name = row['节点名称']
                    if name in pos:
                        x, y = pos[name]
                        node_x.append(x)
                        node_y.append(y)
                        pressure_values.append(row['压力(m)'])
                        node_text.append(f"{name}<br>{row['压力(m)']:.1f}m")
                        node_hover.append(f"节点: {name}<br>压力: {row['压力(m)']:.2f} m<br>水头: {row['水头(m)']:.2f} m")

                fig.add_trace(go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=22,
                        color=pressure_values,
                        colorscale='RdYlGn',
                        cmin=p_min,
                        cmax=p_max,
                        colorbar=dict(
                            title="压力(m)",
                            thickness=15,
                            x=1.02,
                        ),
                        line=dict(color='white', width=2),
                    ),
                    text=node_text,
                    textposition='bottom center',
                    textfont=dict(size=10),
                    hoverinfo='text',
                    hovertext=node_hover,
                    showlegend=False,
                ))

                fig.update_layout(
                    title="管网压力分布（绿色=高压，红色=低压）<br><span style='font-size:12px;color:gray'>黄色虚线=滞流风险，红色虚线=冲刷风险</span>",
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                               scaleanchor='x', scaleratio=1),
                    height=550,
                    margin=dict(l=20, r=80, t=70, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        x=0,
                        y=1,
                        bgcolor='rgba(255,255,255,0.8)',
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("余氯衰减耦合计算")

    result = st.session_state.hydraulic_results
    if result is None:
        st.info('请先在"水力计算"页签中完成水力计算')
    else:
        st.markdown("""
        **余氯衰减模型：**
        
        $C_{下游} = C_{上游} \\cdot e^{-k \\cdot t}$
        
        其中：k为衰减常数(1/h)，t为水力停留时间(h)
        """)

        col_wq1, col_wq2 = st.columns(2)
        with col_wq1:
            k_from_module = st.session_state.get('chlorine_decay_k', None)
            if k_from_module is not None:
                st.info(f"已从水质衰减模块获取衰减常数: k = {k_from_module:.4f} /h")
                k_decay = st.number_input(
                    "衰减常数 k (1/h)",
                    min_value=0.001,
                    value=float(k_from_module),
                    step=0.01,
                    key="k_decay_input",
                )
            else:
                st.info("💡 可在'余氯衰减与CT值计算'模块中计算当月衰减常数")
                k_decay = st.number_input(
                    "衰减常数 k (1/h)",
                    min_value=0.001,
                    value=0.5,
                    step=0.01,
                    help="默认值0.5/h，可在余氯衰减模块中计算当月衰减常数",
                    key="k_decay_default",
                )
        with col_wq2:
            source_chlorine = st.number_input(
                "水源初始余氯浓度 (mg/L)",
                min_value=0.1,
                value=1.0,
                step=0.1,
            )

        if st.button("🧪 计算余氯衰减", type="primary", use_container_width=True):
            with st.spinner("正在计算管网余氯衰减..."):
                wq_result = calculate_water_quality(
                    nodes_df, pipes_df, result,
                    k_decay=k_decay,
                    source_chlorine=source_chlorine,
                )
                st.session_state.water_quality_results = wq_result

        wq = st.session_state.water_quality_results
        if wq is not None:
            st.divider()
            st.subheader("节点余氯浓度")

            node_cl_df = pd.DataFrame(wq['node_chlorine'])
            st.dataframe(node_cl_df, use_container_width=True, hide_index=True)

            warning_nodes = [nr for nr in wq['node_chlorine'] if nr['余氯浓度(mg/L)'] < 0.05]
            if warning_nodes:
                st.error(f"⚠️ 以下节点余氯低于0.05 mg/L: {', '.join(n['节点名称'] for n in warning_nodes)}")
            else:
                st.success("✅ 所有节点余氯浓度均达标 (≥0.05 mg/L)")

            st.divider()
            st.subheader("管段余氯衰减")

            pipe_cl_df = pd.DataFrame(wq['pipe_chlorine'])
            st.dataframe(pipe_cl_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("管网余氯浓度分布图")

            pos = compute_network_layout(nodes_df, pipes_df)
            node_cl_map = {nr['节点名称']: nr['余氯浓度(mg/L)'] for nr in wq['node_chlorine']}

            fig = go.Figure()

            cl_values = list(node_cl_map.values())
            cl_min = min(cl_values) if cl_values else 0
            cl_max = max(cl_values) if cl_values else 1

            diameters = pipes_df['管径(mm)'].values
            d_min = min(diameters)
            d_max = max(diameters)

            for _, row in pipes_df.iterrows():
                start = row['起始节点']
                end = row['终止节点']
                if start in pos and end in pos:
                    x0, y0 = pos[start]
                    x1, y1 = pos[end]
                    d = row['管径(mm)']
                    if d_max > d_min:
                        line_width = 1.5 + 5 * (d - d_min) / (d_max - d_min)
                    else:
                        line_width = 3
                    fig.add_trace(go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines',
                        line=dict(color='#A0C4E8', width=line_width),
                        hoverinfo='text',
                        hovertext=f"{start} → {end}<br>管径: {d} mm",
                        showlegend=False,
                    ))

            normal_x, normal_y, normal_cl, normal_text, normal_hover = [], [], [], [], []
            warning_x, warning_y, warning_cl, warning_text, warning_hover = [], [], [], [], []

            for _, row in nodes_df.iterrows():
                name = row['节点名称']
                if name in pos:
                    x, y = pos[name]
                    cl = node_cl_map.get(name, 0)
                    is_warning = cl < 0.05

                    if is_warning:
                        warning_x.append(x)
                        warning_y.append(y)
                        warning_cl.append(cl)
                        warning_text.append(f"{name}<br>{cl:.3f}mg/L")
                        warning_hover.append(f"节点: {name}<br>余氯: {cl:.4f} mg/L<br>⚠️ 低于0.05 mg/L")
                    else:
                        normal_x.append(x)
                        normal_y.append(y)
                        normal_cl.append(cl)
                        normal_text.append(f"{name}<br>{cl:.3f}mg/L")
                        normal_hover.append(f"节点: {name}<br>余氯: {cl:.4f} mg/L")

            if normal_x:
                fig.add_trace(go.Scatter(
                    x=normal_x,
                    y=normal_y,
                    mode='markers+text',
                    marker=dict(
                        size=18,
                        color=normal_cl,
                        colorscale='YlGnBu',
                        cmin=cl_min,
                        cmax=cl_max,
                        colorbar=dict(
                            title="余氯(mg/L)",
                            thickness=15,
                            x=1.02,
                        ),
                        line=dict(color='white', width=2),
                    ),
                    text=normal_text,
                    textposition='bottom center',
                    textfont=dict(size=10),
                    hoverinfo='text',
                    hovertext=normal_hover,
                    showlegend=False,
                    name='正常节点',
                ))

            if warning_x:
                fig.add_trace(go.Scatter(
                    x=warning_x,
                    y=warning_y,
                    mode='markers+text',
                    marker=dict(
                        size=24,
                        color='red',
                        line=dict(color='white', width=2),
                    ),
                    text=warning_text,
                    textposition='bottom center',
                    textfont=dict(size=10, color='red'),
                    hoverinfo='text',
                    hovertext=warning_hover,
                    showlegend=True,
                    name='预警节点 (<0.05 mg/L)',
                ))

            fig.update_layout(
                title="管网余氯浓度分布（红色=低于0.05mg/L预警）",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           scaleanchor='x', scaleratio=1),
                height=550,
                margin=dict(l=20, r=80, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255,255,255,0.8)',
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("敏感性分析")

    result = st.session_state.hydraulic_results
    if result is None:
        st.info('请先在"水力计算"页签中完成水力计算')
    else:
        st.markdown("选择一个参数进行扫描，分析目标指标随参数变化的趋势")

        nodes_df = st.session_state.network_nodes
        pipes_df = st.session_state.network_pipes

        col_sa1, col_sa2 = st.columns(2)
        with col_sa1:
            param_type = st.selectbox("扫描参数类型", ["管径", "粗糙系数", "用水量"])

            param_id = None
            param_original = None
            if param_type == "管径":
                pipe_options = [
                    f"{idx}: {row['起始节点']}→{row['终止节点']} (DN{int(row['管径(mm)'])})"
                    for idx, row in pipes_df.iterrows()
                ]
                selected_pipe = st.selectbox("选择管段", pipe_options)
                param_id = int(selected_pipe.split(":")[0])
                param_original = pipes_df.loc[param_id, '管径(mm)']
            elif param_type == "粗糙系数":
                pipe_options = [
                    f"{idx}: {row['起始节点']}→{row['终止节点']} (C={int(row['粗糙系数'])})"
                    for idx, row in pipes_df.iterrows()
                ]
                selected_pipe = st.selectbox("选择管段", pipe_options)
                param_id = int(selected_pipe.split(":")[0])
                param_original = pipes_df.loc[param_id, '粗糙系数']
            elif param_type == "用水量":
                demand_nodes = nodes_df[nodes_df['类型'] == '需求']
                node_options = [
                    f"{row['节点名称']} ({row['用水量(m³/h)']}m³/h)"
                    for _, row in demand_nodes.iterrows()
                ]
                if node_options:
                    selected_node = st.selectbox("选择节点", node_options)
                    param_id = selected_node.split(" ")[0]
                    param_original = demand_nodes[demand_nodes['节点名称'] == param_id]['用水量(m³/h)'].values[0]
                else:
                    st.warning("没有需求节点")
                    param_id = None
                    param_original = 0

            range_pct = st.slider("变化范围 (%)", min_value=5, max_value=50, value=20, step=5)
            step_pct = st.slider("步长 (%)", min_value=1, max_value=10, value=5)

            if param_type == "用水量" and param_original == 0:
                param_range = np.array([])
                st.warning('所选节点用水量为 0，无法进行敏感性分析。请选择用水量大于 0 的需求节点，或先在管网拓扑定义中为该节点设置用水量。')
            elif param_original and param_original > 0:
                p_low = param_original * (1 - range_pct / 100.0)
                p_high = param_original * (1 + range_pct / 100.0)
                step = param_original * step_pct / 100.0
                n_steps = int((p_high - p_low) / step) + 1
                param_range = np.linspace(p_low, p_high, n_steps)
                st.caption(f"参数范围: {p_low:.2f} ~ {p_high:.2f}，共 {len(param_range)} 个计算点")
            else:
                param_range = np.array([])
                st.warning("参数原值无效")

        with col_sa2:
            target_type = st.selectbox("目标指标", ["压力", "余氯"])

            if target_type == "压力":
                target_options = nodes_df['节点名称'].tolist()
            else:
                target_options = nodes_df['节点名称'].tolist()
            target_id = st.selectbox("目标节点", target_options)

            if target_type == "余氯":
                st.divider()
                st.markdown("**余氯计算参数:**")
                k_sa = st.session_state.get('chlorine_decay_k', 0.5)
                st.info(f"使用衰减常数: k = {k_sa:.4f} /h")
                source_cl_sa = st.number_input("水源初始余氯 (mg/L)", min_value=0.1, value=1.0, step=0.1, key="sa_source_cl")
            else:
                k_sa = None
                source_cl_sa = 1.0

        if st.button("📊 执行敏感性分析", type="primary", use_container_width=True):
            if len(param_range) == 0:
                st.warning("请先选择有效的参数")
            else:
                with st.spinner("正在执行敏感性分析..."):
                    sa_results = sensitivity_analysis(
                        nodes_df, pipes_df,
                        param_type, param_id, param_range,
                        target_type, target_id,
                        k_decay=k_sa,
                        source_chlorine=source_cl_sa,
                    )

                    if sa_results:
                        sa_df = pd.DataFrame(sa_results)
                        sa_df['参数值'] = sa_df['参数值'].round(4)
                        sa_df['目标值'] = sa_df['目标值'].round(4)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=sa_df['参数值'],
                            y=sa_df['目标值'],
                            mode='lines+markers',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6),
                            name=f'{target_type}({target_id})',
                        ))

                        original_idx = (sa_df['参数值'] - param_original).abs().idxmin()
                        fig.add_vline(
                            x=param_original,
                            line_dash='dash',
                            line_color='red',
                            annotation_text=f'原值: {param_original:.2f}',
                        )

                        unit = "m" if target_type == "压力" else "mg/L"
                        param_unit = ""
                        if param_type == "管径":
                            param_unit = "mm"
                        elif param_type == "粗糙系数":
                            param_unit = ""
                        elif param_type == "用水量":
                            param_unit = "m³/h"

                        fig.update_layout(
                            title=f"{target_type}({target_id}) 随 {param_type} 变化的敏感性分析",
                            xaxis_title=f"{param_type} ({param_unit})",
                            yaxis_title=f"{target_type} ({unit})",
                            height=450,
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(sa_df, use_container_width=True, hide_index=True)

                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("参数原值", f"{param_original:.2f} {param_unit}")
                        with col_m2:
                            original_target = sa_df.loc[original_idx, '目标值'] if original_idx in sa_df.index else sa_df.iloc[len(sa_df)//2]['目标值']
                            st.metric(f"原值对应{target_type}", f"{original_target:.4f} {unit}")
                        with col_m3:
                            sensitivity = abs(sa_df['目标值'].max() - sa_df['目标值'].min())
                            st.metric("目标指标变化幅度", f"{sensitivity:.4f} {unit}")
                    else:
                        st.warning("敏感性分析未产生有效结果")

with tab5:
    st.subheader("管径优化调度 (遗传算法)")

    nodes_df = st.session_state.network_nodes
    pipes_df = st.session_state.network_pipes
    hydraulic_results = st.session_state.hydraulic_results

    if hydraulic_results is None:
        st.info('⚠️ 请先在"水力计算"页签中完成水力计算，再进行管径优化')
    elif len(nodes_df) == 0 or len(pipes_df) == 0:
        st.info('请先在"管网拓扑定义"页签中配置管网结构')
    else:
        errors = validate_network(nodes_df, pipes_df)
        if errors:
            for e in errors:
                st.error(e)
            st.warning("请修正以上错误后再进行优化")
        else:
            st.markdown("""
            **优化目标：** 在满足所有需求节点最低压力约束的前提下，找到管材总费用最低的管径组合方案。

            **候选管径：** [100, 150, 200, 250, 300, 350, 400, 450, 500] mm

            **费用公式：** 每米费用 = 0.5 × 管径(mm) 元/m
            """)

            st.divider()
            st.markdown("##### 优化参数设置")

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                min_pressure = st.number_input(
                    "最低压力阈值 (m)",
                    min_value=0.0,
                    value=15.0,
                    step=1.0,
                    help="所有需求节点的水力计算压力必须不低于此值",
                )
                pop_size = st.number_input(
                    "种群大小",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                )
                max_generations = st.number_input(
                    "最大迭代代数",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                )
            with col_opt2:
                crossover_rate = st.slider(
                    "交叉率",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                )
                mutation_rate = st.slider(
                    "变异率",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                )

            st.divider()

            if st.button("🚀 开始优化", type="primary", use_container_width=True):
                original_cost = calculate_total_cost(pipes_df)
                st.session_state.original_pipe_cost = original_cost

                progress_bar = st.progress(0, text="正在初始化种群...")
                status_text = st.empty()

                def update_progress(gen, best_cost):
                    pct = gen / max_generations
                    progress_bar.progress(pct, text=f"第 {gen}/{max_generations} 代")
                    if best_cost < float('inf'):
                        status_text.info(f"当前最优费用: {best_cost:,.2f} 元")

                with st.spinner("遗传算法优化进行中..."):
                    opt_result = genetic_algorithm_optimization(
                        nodes_df, pipes_df,
                        min_pressure=min_pressure,
                        pop_size=int(pop_size),
                        max_generations=int(max_generations),
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate,
                        progress_callback=update_progress,
                    )
                    st.session_state.optimization_results = opt_result

                    if opt_result['best_cost'] < float('inf'):
                        hc_res = opt_result['best_hydraulic_result']
                        opt_min_pressure = 0.0
                        if hc_res is not None:
                            demand_nr = [nr for nr in hc_res['node_results'] if nr['类型'] == '需求']
                            if demand_nr:
                                opt_min_pressure = min(nr['压力(m)'] for nr in demand_nr)

                        history_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'best_cost': opt_result['best_cost'],
                            'original_cost': original_cost,
                            'saving_pct': ((original_cost - opt_result['best_cost']) / original_cost * 100) if original_cost > 0 else 0,
                            'min_pressure': opt_min_pressure,
                            'best_pipes': opt_result['best_pipes'].copy(),
                            'min_pressure_threshold': min_pressure,
                            'pop_size': int(pop_size),
                            'max_generations': int(max_generations),
                        }
                        st.session_state.optimization_history.append(history_entry)
                        if len(st.session_state.optimization_history) > 5:
                            st.session_state.optimization_history = st.session_state.optimization_history[-5:]

                    progress_bar.progress(1.0, text="优化完成!")
                    status_text.success("✅ 优化完成!")
                    st.rerun()

            opt_result = st.session_state.optimization_results
            if opt_result is not None:
                st.divider()

                optimization_history = st.session_state.get('optimization_history', [])
                if optimization_history:
                    with st.expander("📜 历史方案对比", expanded=False):
                        summary_rows = []
                        for i, entry in enumerate(optimization_history):
                            summary_rows.append({
                                '方案序号': i + 1,
                                '优化时间': entry['timestamp'],
                                '总费用(元)': f"{entry['best_cost']:,.2f}",
                                '节约比例(%)': f"-{entry['saving_pct']:.2f}%",
                                '最低节点压力(m)': f"{entry['min_pressure']:.3f}",
                                '种群大小': entry['pop_size'],
                                '迭代代数': entry['max_generations'],
                            })
                        summary_df = pd.DataFrame(summary_rows)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)

                        for i, entry in enumerate(optimization_history):
                            with st.expander(f"方案 {i + 1} - 详细管径方案 (费用: {entry['best_cost']:,.2f}元)", expanded=False):
                                bp = entry['best_pipes']
                                detail_rows = []
                                for j in range(len(bp)):
                                    orig_d = pipes_df.loc[j, '管径(mm)']
                                    opt_d = bp.loc[j, '管径(mm)']
                                    detail_rows.append({
                                        '管段': f"{pipes_df.loc[j, '起始节点']}→{pipes_df.loc[j, '终止节点']}",
                                        '优化前管径(mm)': int(orig_d),
                                        '优化后管径(mm)': int(opt_d),
                                        '管长(m)': bp.loc[j, '管长(m)'],
                                    })
                                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

                st.subheader("📊 优化结果")

                original_cost = st.session_state.get('original_pipe_cost', calculate_total_cost(pipes_df))
                optimized_cost = opt_result['best_cost']

                if optimized_cost == float('inf'):
                    st.error("❌ 未找到满足压力约束的可行解，请尝试增大管径候选范围或降低最低压力阈值")
                else:
                    col_cost1, col_cost2, col_cost3 = st.columns(3)
                    with col_cost1:
                        st.metric("优化前总费用", f"{original_cost:,.2f} 元")
                    with col_cost2:
                        st.metric("优化后总费用", f"{optimized_cost:,.2f} 元")
                    with col_cost3:
                        saving = original_cost - optimized_cost
                        saving_pct = (saving / original_cost * 100) if original_cost > 0 else 0
                        st.metric("节约费用", f"{saving:,.2f} 元", f"-{saving_pct:.2f}%", delta_color="inverse")

                    st.divider()
                    st.subheader("📋 管径方案对比")

                    best_pipes = opt_result['best_pipes']
                    comparison_data = []
                    for i in range(len(pipes_df)):
                        orig_d = pipes_df.loc[i, '管径(mm)']
                        opt_d = best_pipes.loc[i, '管径(mm)']
                        length = pipes_df.loc[i, '管长(m)']
                        unit_price = 0.5 * opt_d
                        cost = calculate_pipe_cost(opt_d, length)
                        orig_cost = calculate_pipe_cost(orig_d, length)

                        if opt_d > orig_d:
                            change = "增大"
                        elif opt_d < orig_d:
                            change = "减小"
                        else:
                            change = "不变"

                        comparison_data.append({
                            '管段': f"{pipes_df.loc[i, '起始节点']}→{pipes_df.loc[i, '终止节点']}",
                            '优化前管径(mm)': int(orig_d),
                            '优化后管径(mm)': int(opt_d),
                            '管径变化': change,
                            '管长(m)': length,
                            '单价(元/m)': unit_price,
                            '优化前费用(元)': round(orig_cost, 2),
                            '优化后费用(元)': round(cost, 2),
                            '费用变化(元)': round(cost - orig_cost, 2),
                        })

                    comp_df = pd.DataFrame(comparison_data)

                    def highlight_change(row):
                        if row['管径变化'] == '增大':
                            return ['background-color: #f8d7da; color: #721c24'] * len(row)
                        elif row['管径变化'] == '减小':
                            return ['background-color: #d4edda; color: #155724'] * len(row)
                        else:
                            return [''] * len(row)

                    styled_comp = comp_df.style.apply(highlight_change, axis=1)
                    st.dataframe(styled_comp, use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("📈 收敛曲线")

                    cost_history = opt_result['cost_history']
                    valid_costs = [c for c in cost_history if c < float('inf')]
                    generations = list(range(1, len(cost_history) + 1))

                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=generations,
                        y=cost_history,
                        mode='lines+markers',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4),
                        name='最优费用',
                    ))
                    fig_conv.update_layout(
                        title="遗传算法收敛曲线",
                        xaxis_title="迭代代数",
                        yaxis_title="最优管材总费用 (元)",
                        height=400,
                        hovermode='x unified',
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)

                    st.divider()
                    st.subheader("🗺️ 管径变化拓扑图")

                    pos = compute_network_layout(nodes_df, pipes_df)

                    fig_opt = go.Figure()

                    for i, row in pipes_df.iterrows():
                        start = row['起始节点']
                        end = row['终止节点']
                        if start in pos and end in pos:
                            x0, y0 = pos[start]
                            x1, y1 = pos[end]
                            orig_d = row['管径(mm)']
                            opt_d = best_pipes.loc[i, '管径(mm)']

                            if opt_d > orig_d:
                                line_color = '#DC3545'
                                status_text = '管径增大'
                            elif opt_d < orig_d:
                                line_color = '#28A745'
                                status_text = '管径减小'
                            else:
                                line_color = '#6C757D'
                                status_text = '管径不变'

                            line_width = 3 + (opt_d / 500) * 5

                            fig_opt.add_trace(go.Scatter(
                                x=[x0, x1],
                                y=[y0, y1],
                                mode='lines',
                                line=dict(color=line_color, width=line_width),
                                hoverinfo='text',
                                hovertext=(
                                    f"{start}→{end}<br>"
                                    f"优化前: DN{int(orig_d)}mm<br>"
                                    f"优化后: DN{int(opt_d)}mm<br>"
                                    f"状态: {status_text}"
                                ),
                                showlegend=False,
                            ))

                    legend_items = [
                        ('管径增大', '#DC3545', 'line'),
                        ('管径减小', '#28A745', 'line'),
                        ('管径不变', '#6C757D', 'line'),
                    ]
                    for name, color, mode in legend_items:
                        fig_opt.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='lines',
                            line=dict(color=color, width=4),
                            name=name,
                            showlegend=True,
                        ))

                    for _, row in nodes_df.iterrows():
                        name = row['节点名称']
                        if name in pos:
                            x, y = pos[name]
                            is_source = row['类型'] == '水源'
                            color = 'blue' if is_source else 'green'
                            size = 16

                            label = f"{name}"
                            if is_source:
                                label += f"<br>水源"

                            fig_opt.add_trace(go.Scatter(
                                x=[x],
                                y=[y],
                                mode='markers+text',
                                marker=dict(size=size, color=color,
                                            line=dict(color='white', width=2)),
                                text=[label],
                                textposition='bottom center',
                                textfont=dict(size=10),
                                hoverinfo='text',
                                showlegend=False,
                            ))

                    fig_opt.update_layout(
                        title="管网管径变化分布图",
                        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                                   scaleanchor='x', scaleratio=1),
                        height=550,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        legend=dict(
                            x=0,
                            y=1,
                            bgcolor='rgba(255,255,255,0.8)',
                        ),
                    )
                    st.plotly_chart(fig_opt, use_container_width=True)

                    st.divider()
                    st.subheader("💧 优化后水力校核")

                    hc_result = opt_result['best_hydraulic_result']
                    if hc_result is not None:
                        if hc_result['converged']:
                            st.success(f"✅ 水力校核收敛，共迭代 {hc_result['iterations']} 次")
                        else:
                            st.warning("⚠️ 水力校核未收敛")

                        node_res_df = pd.DataFrame(hc_result['node_results'])
                        demand_nodes = node_res_df[node_res_df['类型'] == '需求']

                        if not demand_nodes.empty:
                            min_p = demand_nodes['压力(m)'].min()
                            min_p_node = demand_nodes.loc[demand_nodes['压力(m)'].idxmin(), '节点名称']

                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                st.metric("最低压力节点", min_p_node)
                            with col_p2:
                                st.metric("最低压力值", f"{min_p:.3f} m", delta=f"≥{min_pressure}m")

                            if min_p >= min_pressure:
                                st.success("✅ 所有需求节点压力均满足约束")
                            else:
                                st.error(f"❌ 节点 {min_p_node} 压力不满足约束 ({min_p:.3f}m < {min_pressure}m)")

                        display_node = node_res_df[['节点名称', '类型', '水头(m)', '压力(m)', '用水量(m³/h)']].copy()
                        display_node['水头(m)'] = display_node['水头(m)'].round(3)
                        display_node['压力(m)'] = display_node['压力(m)'].round(3)
                        st.dataframe(display_node, use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("📏 约束余量分析")

                    if hc_result is not None:
                        node_res_df = pd.DataFrame(hc_result['node_results'])
                        demand_nodes = node_res_df[node_res_df['类型'] == '需求'].copy()

                        if not demand_nodes.empty:
                            demand_nodes['压力余量(m)'] = demand_nodes['压力(m)'] - min_pressure
                            demand_nodes = demand_nodes.sort_values('压力余量(m)', ascending=True)

                            bottleneck_nodes = demand_nodes.nsmallest(3, '压力余量(m)')['节点名称'].tolist()
                            bottleneck_set = set(bottleneck_nodes)

                            fig_margin = go.Figure()

                            colors = ['#DC3545' if name in bottleneck_set else '#4A90D9' for name in demand_nodes['节点名称']]

                            fig_margin.add_trace(go.Bar(
                                y=demand_nodes['节点名称'].tolist(),
                                x=demand_nodes['压力余量(m)'].tolist(),
                                orientation='h',
                                marker_color=colors,
                                text=[f"{v:.3f}" for v in demand_nodes['压力余量(m)']],
                                textposition='auto',
                                hovertemplate='节点: %{y}<br>压力余量: %{x:.3f} m<extra></extra>',
                                showlegend=False,
                            ))

                            fig_margin.add_vline(
                                x=0,
                                line_dash='dash',
                                line_color='green',
                                line_width=2,
                                annotation_text="余量=0",
                                annotation_position="top right",
                            )

                            fig_margin.add_trace(go.Scatter(
                                x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=10, color='#DC3545'),
                                name='瓶颈节点 (余量最小Top3)',
                                showlegend=True,
                            ))
                            fig_margin.add_trace(go.Scatter(
                                x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=10, color='#4A90D9'),
                                name='其他需求节点',
                                showlegend=True,
                            ))

                            fig_margin.update_layout(
                                title="需求节点压力余量分布（从小到大排列）",
                                xaxis_title="压力余量 (m)",
                                yaxis_title="节点名称",
                                height=max(300, len(demand_nodes) * 35 + 100),
                                margin=dict(l=80, r=30, t=60, b=40),
                                legend=dict(
                                    x=0.7,
                                    y=1.1,
                                    orientation='h',
                                ),
                            )
                            st.plotly_chart(fig_margin, use_container_width=True)

                            st.markdown(f"**瓶颈节点（压力余量最小的3个）:** {', '.join(bottleneck_nodes)}")
                            for _, row in demand_nodes.iterrows():
                                if row['节点名称'] in bottleneck_set:
                                    margin_val = row['压力余量(m)']
                                    if margin_val < 0:
                                        st.error(f"❌ {row['节点名称']}: 压力余量 {margin_val:.3f}m，未满足约束")
                                    elif margin_val < 2:
                                        st.warning(f"⚠️ {row['节点名称']}: 压力余量仅 {margin_val:.3f}m，接近约束边界")
                                    else:
                                        st.info(f"📌 {row['节点名称']}: 压力余量 {margin_val:.3f}m")

                    st.divider()
                    st.subheader("🧪 水质影响评估")

                    if hc_result is not None:
                        pre_wq = st.session_state.water_quality_results
                        k_decay_for_opt = st.session_state.get('chlorine_decay_k', 0.5)
                        source_chlorine_for_opt = 1.0

                        best_pipes = opt_result['best_pipes']
                        post_wq = calculate_water_quality(
                            nodes_df, best_pipes, hc_result,
                            k_decay=k_decay_for_opt,
                            source_chlorine=source_chlorine_for_opt,
                        )

                        if pre_wq is not None:
                            pre_cl_map = {nr['节点名称']: nr['余氯浓度(mg/L)'] for nr in pre_wq['node_chlorine']}
                        else:
                            pre_hydraulic = st.session_state.hydraulic_results
                            if pre_hydraulic is not None:
                                pre_wq_calc = calculate_water_quality(
                                    nodes_df, pipes_df, pre_hydraulic,
                                    k_decay=k_decay_for_opt,
                                    source_chlorine=source_chlorine_for_opt,
                                )
                                pre_cl_map = {nr['节点名称']: nr['余氯浓度(mg/L)'] for nr in pre_wq_calc['node_chlorine']}
                            else:
                                pre_cl_map = {}

                        post_cl_map = {nr['节点名称']: nr['余氯浓度(mg/L)'] for nr in post_wq['node_chlorine']}

                        wq_compare_rows = []
                        for _, row in nodes_df.iterrows():
                            name = row['节点名称']
                            pre_cl = pre_cl_map.get(name, 0.0)
                            post_cl = post_cl_map.get(name, 0.0)
                            delta = post_cl - pre_cl
                            risk = "⚠️ 不达标" if (row['类型'] == '需求' and post_cl < 0.05) else ""
                            wq_compare_rows.append({
                                '节点名称': name,
                                '类型': row['类型'],
                                '优化前余氯(mg/L)': round(pre_cl, 6),
                                '优化后余氯(mg/L)': round(post_cl, 6),
                                '变化量(mg/L)': round(delta, 6),
                                '状态': risk,
                            })

                        wq_compare_df = pd.DataFrame(wq_compare_rows)

                        def highlight_risk(row):
                            if row['状态'] == "⚠️ 不达标":
                                return ['background-color: #f8d7da; color: #721c24'] * len(row)
                            return [''] * len(row)

                        styled_wq = wq_compare_df.style.apply(highlight_risk, axis=1)
                        st.dataframe(styled_wq, use_container_width=True, hide_index=True)

                        at_risk_nodes = [r for r in wq_compare_rows if r['状态'] == "⚠️ 不达标"]
                        if at_risk_nodes:
                            for r in at_risk_nodes:
                                st.error(f"⚠️ 管径优化后该节点存在余氯不达标风险，建议重新评估: {r['节点名称']} (余氯: {r['优化后余氯(mg/L)']:.4f} mg/L < 0.05 mg/L)")
                        else:
                            st.success("✅ 管径优化后所有需求节点余氯浓度均达标 (≥0.05 mg/L)")
