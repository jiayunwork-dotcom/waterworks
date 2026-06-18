import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network_utils import (
    validate_network,
    hardy_cross_calculation,
    calculate_water_quality,
    sensitivity_analysis,
    compute_network_layout,
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

tab1, tab2, tab3, tab4 = st.tabs(["📐 管网拓扑定义", "💧 水力计算", "🧪 水质衰减耦合", "📊 敏感性分析"])

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
                        (existing['起始节点'] == new_pipe_start) & (existing['终止节点'] == new_pipe_end)
                    ]
                    if not dup.empty:
                        st.warning(f"管段 {new_pipe_start} → {new_pipe_end} 已存在")
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
                    symbol = 'diamond' if is_source else 'circle'
                    size = 18 if is_source else 14

                    label = f"{name}"
                    if is_source:
                        label += f"<br>水源 H={row['水头(m)']}m"
                    else:
                        label += f"<br>需求 {row['用水量(m³/h)']}m³/h"

                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(size=size, color=color, symbol=symbol,
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
                    display_pipe = pipe_df[['起始节点', '终止节点', '管径(mm)', '管长(m)',
                                             '流量(L/s)', '流速(m/s)', '水头损失(m)']].copy()
                    display_pipe['流量(L/s)'] = display_pipe['流量(L/s)'].round(3)
                    display_pipe['流速(m/s)'] = display_pipe['流速(m/s)'].round(4)
                    display_pipe['水头损失(m)'] = display_pipe['水头损失(m)'].round(4)
                    st.dataframe(display_pipe, use_container_width=True, hide_index=True)

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
                            line=dict(color='#888888', width=line_width),
                            hoverinfo='text',
                            hovertext=f"{start} → {end}<br>管径: {d} mm",
                            showlegend=False,
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
                    title="管网压力分布（绿色=高压，红色=低压）",
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                               scaleanchor='x', scaleratio=1),
                    height=550,
                    margin=dict(l=20, r=80, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
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

            if param_original and param_original > 0:
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
