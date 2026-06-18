import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime


def generate_report(
    df,
    report_type='日报',
    start_date=None,
    end_date=None,
    include_sections=None,
    energy_df=None,
):
    if include_sections is None:
        include_sections = [
            '核心指标统计',
            '超标事件汇总',
            '投药量对比',
            'CT值达标情况',
            '能耗摘要',
        ]
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=10*mm,
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=8*mm,
        spaceAfter=4*mm,
        textColor=colors.HexColor('#1f77b4'),
    )
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=2*mm,
    )
    
    story = []
    
    title_text = f"自来水厂水质{report_type}"
    if start_date and end_date:
        title_text += f"\n{start_date} 至 {end_date}"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 5*mm))
    
    if '核心指标统计' in include_sections:
        story.append(Paragraph("一、核心指标统计", h2_style))
        stats_data = generate_core_stats_table(df)
        if stats_data is not None:
            t = Table(stats_data, colWidths=[40*mm, 25*mm, 25*mm, 25*mm, 25*mm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
    
    if '超标事件汇总' in include_sections:
        story.append(Paragraph("二、超标事件汇总", h2_style))
        violation_summary = generate_violation_summary(df)
        if violation_summary is not None:
            t = Table(violation_summary, colWidths=[35*mm, 25*mm, 30*mm, 35*mm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d62728')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
    
    if '投药量对比' in include_sections:
        story.append(Paragraph("三、投药量统计", h2_style))
        dosage_stats = generate_dosage_stats(df)
        if dosage_stats is not None:
            t = Table(dosage_stats, colWidths=[40*mm, 30*mm, 30*mm, 30*mm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
    
    if 'CT值达标情况' in include_sections:
        story.append(Paragraph("四、CT值达标情况", h2_style))
        ct_info = generate_ct_info()
        story.append(Paragraph(ct_info, normal_style))
    
    if '能耗摘要' in include_sections and energy_df is not None:
        story.append(Paragraph("五、能耗摘要", h2_style))
        energy_summary = generate_energy_summary_table(energy_df)
        if energy_summary is not None:
            t = Table(energy_summary, colWidths=[40*mm, 30*mm, 30*mm, 30*mm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
    
    doc.build(story)
    buffer.seek(0)
    
    return buffer


def generate_core_stats_table(df):
    if df is None or len(df) == 0:
        return None
    
    key_metrics = [
        ('源水浊度', 'NTU'),
        ('源水pH', ''),
        ('源水温度', '℃'),
        ('混凝剂投加量', 'mg/L'),
        ('沉后浊度', 'NTU'),
        ('滤后浊度', 'NTU'),
        ('出水浊度', 'NTU'),
        ('出水余氯', 'mg/L'),
        ('出水pH', ''),
    ]
    
    data = [['指标', '均值', '最小值', '最大值', '单位']]
    
    for metric, unit in key_metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors='coerce').dropna()
        if len(values) == 0:
            continue
        data.append([
            metric,
            f"{values.mean():.3f}",
            f"{values.min():.3f}",
            f"{values.max():.3f}",
            unit,
        ])
    
    return data if len(data) > 1 else None


def generate_violation_summary(df):
    from utils.data_utils import get_violations
    
    violations_df = get_violations(df)
    if violations_df is None or violations_df.empty:
        return [['指标', '超标次数', '最大超标幅度', '备注'],
                ['-', '0', '-', '无超标事件']]
    
    summary = violations_df.groupby('指标').agg(
        超标次数=('数值', 'count'),
        最大超标幅度=('超标幅度', 'max'),
    ).reset_index()
    
    data = [['指标', '超标次数', '最大超标幅度', '超标类型']]
    for _, row in summary.iterrows():
        data.append([
            row['指标'],
            str(row['超标次数']),
            f"{row['最大超标幅度']:.4f}",
            '超标',
        ])
    
    return data


def generate_dosage_stats(df):
    if '混凝剂投加量' not in df.columns:
        return None
    
    values = pd.to_numeric(df['混凝剂投加量'], errors='coerce').dropna()
    if len(values) == 0:
        return None
    
    data = [
        ['统计项', '数值', '单位', '备注'],
        ['平均投药量', f"{values.mean():.3f}", 'mg/L', ''],
        ['最小投药量', f"{values.min():.3f}", 'mg/L', ''],
        ['最大投药量', f"{values.max():.3f}", 'mg/L', ''],
        ['投药量标准差', f"{values.std():.3f}", 'mg/L', ''],
        ['数据点数', f"{len(values)}", '个', ''],
    ]
    
    return data


def generate_ct_info():
    return ("CT值(余氯浓度×接触时间)是衡量消毒效果的重要指标。\n"
            "国家标准要求: CT ≥ 15 mg·min/L\n"
            "实际CT值需根据现场清水池容积和供水流量计算。")


def generate_energy_summary_table(energy_df):
    if energy_df is None or len(energy_df) == 0:
        return None
    
    energy_cols = [c for c in ['取水泵(kWh)', '搅拌器(kWh)', '反冲洗泵(kWh)', '加压泵(kWh)'] if c in energy_df.columns]
    
    if not energy_cols:
        return None
    
    data = [['能耗环节', '累计能耗(kWh)', '日均能耗(kWh)', '占比(%)']]
    
    total = sum(energy_df[col].sum() for col in energy_cols)
    
    for col in energy_cols:
        val = energy_df[col].sum()
        daily_avg = energy_df[col].mean()
        pct = (val / total * 100) if total > 0 else 0
        data.append([
            col.replace('(kWh)', ''),
            f"{val:.2f}",
            f"{daily_avg:.2f}",
            f"{pct:.2f}%",
        ])
    
    return data
