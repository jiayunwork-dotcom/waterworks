import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.report_utils import generate_report


st.header("📄 报告导出")

if st.session_state.df is None:
    st.info("请先上传数据或加载示例数据。")
else:
    df = st.session_state.df
    energy_df = st.session_state.energy_df
    
    st.subheader("报告配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "报告类型",
            ['日报', '月报', '自定义'],
            index=0,
        )
    
    with col2:
        date_range_option = st.selectbox(
            "日期范围",
            ['最近一天', '最近一周', '最近一月', '全部数据', '自定义'],
            index=2,
        )
    
    if date_range_option == '自定义':
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=df['时间'].min().date())
        with col2:
            end_date = st.date_input("结束日期", value=df['时间'].max().date())
    else:
        end_date = df['时间'].max().date()
        if date_range_option == '最近一天':
            start_date = end_date
        elif date_range_option == '最近一周':
            start_date = end_date - pd.Timedelta(days=7)
        elif date_range_option == '最近一月':
            start_date = end_date - pd.Timedelta(days=30)
        else:
            start_date = df['时间'].min().date()
            end_date = df['时间'].max().date()
    
    st.divider()
    
    st.subheader("包含章节")
    
    include_sections = []
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("核心指标统计", value=True):
            include_sections.append("核心指标统计")
        if st.checkbox("超标事件汇总", value=True):
            include_sections.append("超标事件汇总")
    with col2:
        if st.checkbox("投药量统计", value=True):
            include_sections.append("投药量对比")
        if st.checkbox("CT值达标情况", value=True):
            include_sections.append("CT值达标情况")
    with col3:
        if st.checkbox("能耗摘要", value=True, disabled=energy_df is None):
            include_sections.append("能耗摘要")
    
    st.divider()
    
    report_df = df.copy()
    report_df = report_df[
        (report_df['时间'].dt.date >= start_date) & 
        (report_df['时间'].dt.date <= end_date)
    ]
    
    report_energy_df = None
    if energy_df is not None:
        report_energy_df = energy_df.copy()
        report_energy_df = report_energy_df[
            (report_energy_df['日期'].dt.date >= start_date) & 
            (report_energy_df['日期'].dt.date <= end_date)
        ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("报告期数据点数", len(report_df))
    with col2:
        st.metric("开始日期", start_date.strftime('%Y-%m-%d'))
    with col3:
        st.metric("结束日期", end_date.strftime('%Y-%m-%d'))
    
    st.divider()
    
    if st.button("📄 生成PDF报告", type="primary", use_container_width=True):
        with st.spinner("正在生成报告..."):
            try:
                pdf_buffer = generate_report(
                    report_df,
                    report_type=report_type,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    include_sections=include_sections,
                    energy_df=report_energy_df,
                )
                
                st.success("✅ 报告生成成功！")
                
                report_filename = f"水质{report_type}_{start_date}_{end_date}.pdf"
                
                st.download_button(
                    "📥 下载PDF报告",
                    data=pdf_buffer,
                    file_name=report_filename,
                    mime='application/pdf',
                    use_container_width=True,
                )
                
                st.divider()
                st.subheader("报告预览（文字摘要）")
                
                if "核心指标统计" in include_sections:
                    st.markdown("### 一、核心指标统计")
                    key_metrics = ['源水浊度', '混凝剂投加量', '出水浊度', '出水余氯', '出水pH']
                    for metric in key_metrics:
                        if metric in report_df.columns:
                            values = pd.to_numeric(report_df[metric], errors='coerce').dropna()
                            if len(values) > 0:
                                st.write(f"- **{metric}**: 均值 {values.mean():.3f}, 范围 [{values.min():.3f}, {values.max():.3f}]")
                
                if "超标事件汇总" in include_sections:
                    st.markdown("### 二、超标事件汇总")
                    from utils.data_utils import get_violations
                    violations = get_violations(report_df)
                    if violations is not None and not violations.empty:
                        st.write(f"共发现 {len(violations)} 起超标事件")
                        for idx, row in violations.head(5).iterrows():
                            st.write(f"- {row['时间']}: {row['指标']} {row['超标类型']} ({row['数值']})")
                        if len(violations) > 5:
                            st.write(f"... 还有 {len(violations) - 5} 起")
                    else:
                        st.write("✅ 无超标事件")
                
                if "投药量对比" in include_sections and "混凝剂投加量" in report_df.columns:
                    st.markdown("### 三、投药量统计")
                    dosage = pd.to_numeric(report_df['混凝剂投加量'], errors='coerce').dropna()
                    if len(dosage) > 0:
                        st.write(f"- 平均投药量: {dosage.mean():.3f} mg/L")
                        st.write(f"- 投药量范围: [{dosage.min():.3f}, {dosage.max():.3f}] mg/L")
                
                if "能耗摘要" in include_sections and report_energy_df is not None:
                    st.markdown("### 五、能耗摘要")
                    from utils.energy_utils import calculate_unit_energy, get_energy_breakdown
                    energy_processed, energy_cols = calculate_unit_energy(report_energy_df)
                    breakdown = get_energy_breakdown(energy_processed, energy_cols)
                    if not breakdown.empty:
                        total = breakdown['能耗(kWh)'].sum()
                        st.write(f"- 总能耗: {total:,.2f} kWh")
                        for _, row in breakdown.iterrows():
                            st.write(f"  - {row['环节']}: {row['能耗(kWh)']:,.2f} kWh ({row['占比(%)']}%)")
                
            except Exception as e:
                st.error(f"报告生成失败: {e}")
    
    st.divider()
    st.info("💡 提示：PDF报告包含统计数据表格，可直接打印或存档使用。")
