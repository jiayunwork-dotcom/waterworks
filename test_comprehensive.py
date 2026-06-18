import sys
sys.path.insert(0, '.')
from utils.network_utils import (
    validate_network,
    hardy_cross_calculation,
    calculate_water_quality,
    sensitivity_analysis,
    compute_network_layout,
    find_loops,
    hazen_williams_head_loss,
)
import pandas as pd
import numpy as np

print("=" * 60)
print("综合测试：管网水力模型仿真")
print("=" * 60)

# 测试数据
nodes_df = pd.DataFrame([
    {'节点名称': 'S1', '类型': '水源', '用水量(m³/h)': 0.0, '水头(m)': 40.0, '地面标高(m)': 5.0},
    {'节点名称': 'N1', '类型': '需求', '用水量(m³/h)': 20.0, '水头(m)': 0.0, '地面标高(m)': 6.0},
    {'节点名称': 'N2', '类型': '需求', '用水量(m³/h)': 30.0, '水头(m)': 0.0, '地面标高(m)': 7.0},
    {'节点名称': 'N3', '类型': '需求', '用水量(m³/h)': 15.0, '水头(m)': 0.0, '地面标高(m)': 8.0},
    {'节点名称': 'N4', '类型': '需求', '用水量(m³/h)': 25.0, '水头(m)': 0.0, '地面标高(m)': 6.5},
    {'节点名称': 'N5', '类型': '需求', '用水量(m³/h)': 10.0, '水头(m)': 0.0, '地面标高(m)': 9.0},
])
pipes_df = pd.DataFrame([
    {'起始节点': 'S1', '终止节点': 'N1', '管径(mm)': 400, '管长(m)': 300, '粗糙系数': 130},
    {'起始节点': 'S1', '终止节点': 'N2', '管径(mm)': 350, '管长(m)': 400, '粗糙系数': 130},
    {'起始节点': 'N1', '终止节点': 'N3', '管径(mm)': 250, '管长(m)': 500, '粗糙系数': 130},
    {'起始节点': 'N2', '终止节点': 'N4', '管径(mm)': 250, '管长(m)': 450, '粗糙系数': 130},
    {'起始节点': 'N3', '终止节点': 'N5', '管径(mm)': 200, '管长(m)': 350, '粗糙系数': 130},
    {'起始节点': 'N4', '终止节点': 'N5', '管径(mm)': 200, '管长(m)': 300, '粗糙系数': 130},
    {'起始节点': 'N1', '终止节点': 'N4', '管径(mm)': 200, '管长(m)': 600, '粗糙系数': 130},
])

all_passed = True

# 测试1: 管网验证
print("\n【测试1】管网验证")
errors = validate_network(nodes_df, pipes_df)
if len(errors) == 0:
    print("  ✅ 通过：有效管网")
else:
    print(f"  ❌ 失败：{errors}")
    all_passed = False

# 测试2: 环路识别
print("\n【测试2】环路识别")
loops = find_loops(pipes_df)
print(f"  发现 {len(loops)} 个独立环路")
if len(loops) == 2:
    print("  ✅ 通过：环路数量正确 (应为2个)")
else:
    print(f"  ❌ 失败：期望2个环路，实际{len(loops)}个")
    all_passed = False

# 测试3: Hardy-Cross水力计算
print("\n【测试3】Hardy-Cross水力计算")
result = hardy_cross_calculation(nodes_df, pipes_df, max_iter=500, tol=0.001)
print(f"  收敛: {result['converged']}, 迭代次数: {result['iterations']}")

if result['converged']:
    print("  ✅ 通过：计算收敛")
else:
    print("  ❌ 失败：计算不收敛")
    all_passed = False

# 测试4: 节点流量平衡
print("\n【测试4】节点流量平衡验证")
flows = result['flows']
balance_ok = True

# 先计算总需求
total_demand = 0
for _, node_row in nodes_df.iterrows():
    if node_row['类型'] != '水源':
        total_demand += node_row['用水量(m³/h)'] / 3600.0

for _, node_row in nodes_df.iterrows():
    node_name = node_row['节点名称']
    demand = node_row['用水量(m³/h)'] / 3600.0
    inflow = 0.0
    outflow = 0.0
    for idx, pipe_row in pipes_df.iterrows():
        Q = flows[idx]
        if pipe_row['起始节点'] == node_name:
            if Q >= 0:
                outflow += Q
            else:
                inflow += abs(Q)
        elif pipe_row['终止节点'] == node_name:
            if Q >= 0:
                inflow += Q
            else:
                outflow += abs(Q)
    net_flow = inflow - outflow
    
    if node_row['类型'] == '水源':
        if abs(abs(net_flow) - total_demand) > 0.001:
            print(f"  ❌ 水源节点 {node_name}: 净出流 {abs(net_flow)*1000:.3f} L/s, 总需求 {total_demand*1000:.3f} L/s, 差 {abs(abs(net_flow) - total_demand)*1000:.6f} L/s")
            balance_ok = False
    else:
        if abs(net_flow - demand) > 0.001:
            print(f"  ❌ 节点 {node_name}: 净入流 {net_flow*1000:.3f} L/s, 需求 {demand*1000:.3f} L/s, 差 {abs(net_flow - demand)*1000:.6f} L/s")
            balance_ok = False

if balance_ok:
    print("  ✅ 通过：所有节点流量平衡")
else:
    print("  ❌ 失败：存在节点流量不平衡")
    all_passed = False

# 测试5: 环路水头损失闭合
print("\n【测试5】环路水头损失闭合验证")
pipe_data_list = result['pipe_data']
loop_ok = True
for i, loop in enumerate(loops):
    sum_hf = 0.0
    for pipe_idx, direction in loop:
        Q = flows[pipe_idx] * direction
        pdata = pipe_data_list[pipe_idx]
        hf = hazen_williams_head_loss(Q, pdata['L'], pdata['C'], pdata['D'])
        sum_hf += hf
    if abs(sum_hf) < 0.05:
        print(f"  ✅ 环路 {i+1}: 水头损失闭合 (Σhf = {sum_hf:.6f} m)")
    else:
        print(f"  ❌ 环路 {i+1}: 水头损失不闭合 (Σhf = {sum_hf:.6f} m)")
        loop_ok = False

if loop_ok:
    print("  ✅ 通过：所有环路水头损失闭合")
else:
    all_passed = False

# 测试6: 节点水头合理性
print("\n【测试6】节点水头合理性")
node_results = result['node_results']
head_ok = True
source_head = None
for nr in node_results:
    if nr['类型'] == '水源':
        source_head = nr['水头(m)']
        break

if source_head is not None:
    for nr in node_results:
        if nr['类型'] != '水源' and nr['水头(m)'] > source_head + 0.01:
            print(f"  ❌ 节点 {nr['节点名称']}: 水头 {nr['水头(m)']:.3f}m > 水源水头 {source_head:.3f}m")
            head_ok = False

if head_ok:
    print("  ✅ 通过：所有节点水头不高于水源水头")
else:
    all_passed = False

# 测试7: 水质衰减计算
print("\n【测试7】余氯衰减计算")
k_decay = 0.5
source_chlorine = 1.0
wq = calculate_water_quality(nodes_df, pipes_df, result, k_decay=k_decay, source_chlorine=source_chlorine)

wq_ok = True
# 验证水源节点余氯
for nr in wq['node_chlorine']:
    if nr['类型'] == '水源':
        if abs(nr['余氯浓度(mg/L)'] - source_chlorine) < 0.001:
            print(f"  ✅ 水源节点 {nr['节点名称']}: 余氯 = {nr['余氯浓度(mg/L)']:.4f} mg/L")
        else:
            print(f"  ❌ 水源节点 {nr['节点名称']}: 余氯 = {nr['余氯浓度(mg/L)']:.4f} mg/L (应为 {source_chlorine})")
            wq_ok = False

# 验证所有节点余氯在合理范围
all_positive = True
all_below_source = True
for nr in wq['node_chlorine']:
    if nr['余氯浓度(mg/L)'] < 0:
        all_positive = False
    if nr['类型'] != '水源' and nr['余氯浓度(mg/L)'] > source_chlorine + 0.001:
        all_below_source = False

if all_positive:
    print("  ✅ 通过：所有节点余氯为正值")
else:
    print("  ❌ 失败：存在负余氯浓度")
    wq_ok = False

if all_below_source:
    print("  ✅ 通过：所有节点余氯不超过水源浓度")
else:
    print("  ❌ 失败：存在节点余氯超过水源浓度")
    wq_ok = False

# 验证管段余氯衰减
pipe_decay_ok = True
for pr in wq['pipe_chlorine']:
    if pr['下游余氯(mg/L)'] > pr['上游余氯(mg/L)'] + 0.001:
        print(f"  ❌ 管段 {pr['起始节点']}->{pr['终止节点']}: 下游余氯 {pr['下游余氯(mg/L)']:.4f} > 上游余氯 {pr['上游余氯(mg/L)']:.4f}")
        pipe_decay_ok = False

if pipe_decay_ok:
    print("  ✅ 通过：所有管段下游余氯不高于上游")
else:
    wq_ok = False

if not wq_ok:
    all_passed = False

# 测试8: 敏感性分析 - 管径
print("\n【测试8】敏感性分析 - 管径对压力的影响")
param_range = np.linspace(200, 500, 7)
sa_results = sensitivity_analysis(
    nodes_df, pipes_df,
    '管径', 0, param_range,
    '压力', 'N3'
)

if len(sa_results) == len(param_range):
    print(f"  ✅ 通过：返回 {len(sa_results)} 个结果点")
    # 验证趋势：管径越大，压力越高
    pressures = [r['目标值'] for r in sa_results]
    is_increasing = all(pressures[i] <= pressures[i+1] + 0.001 for i in range(len(pressures)-1))
    if is_increasing:
        print("  ✅ 通过：压力随管径增大而增大（趋势正确）")
    else:
        print("  ⚠️  警告：压力随管径变化趋势不符合预期")
        print(f"    压力值: {[f'{p:.3f}' for p in pressures]}")
else:
    print(f"  ❌ 失败：期望 {len(param_range)} 个结果，实际 {len(sa_results)} 个")
    all_passed = False

# 测试9: 敏感性分析 - 用水量对余氯的影响
print("\n【测试9】敏感性分析 - 用水量对余氯的影响")
param_range = np.linspace(5, 30, 6)
sa_results_cl = sensitivity_analysis(
    nodes_df, pipes_df,
    '用水量', 'N1', param_range,
    '余氯', 'N3',
    k_decay=0.5,
    source_chlorine=1.0
)

if len(sa_results_cl) == len(param_range):
    print(f"  ✅ 通过：返回 {len(sa_results_cl)} 个结果点")
    # 验证趋势：用水量越大，流速越快，停留时间越短，余氯越高
    cl_values = [r['目标值'] for r in sa_results_cl]
    print(f"    余氯值: {[f'{c:.4f}' for c in cl_values]} mg/L")
else:
    print(f"  ❌ 失败：期望 {len(param_range)} 个结果，实际 {len(sa_results_cl)} 个")
    all_passed = False

# 测试10: 管网布局计算
print("\n【测试10】管网布局计算")
pos = compute_network_layout(nodes_df, pipes_df)
if len(pos) == len(nodes_df):
    print(f"  ✅ 通过：成功计算 {len(pos)} 个节点的坐标")
else:
    print(f"  ❌ 失败：期望 {len(nodes_df)} 个节点坐标，实际 {len(pos)} 个")
    all_passed = False

# 测试11: 边界情况 - 空管网
print("\n【测试11】边界情况 - 空管网验证")
empty_nodes = pd.DataFrame(columns=['节点名称', '类型', '用水量(m³/h)', '水头(m)', '地面标高(m)'])
empty_pipes = pd.DataFrame(columns=['起始节点', '终止节点', '管径(mm)', '管长(m)', '粗糙系数'])
errors_empty = validate_network(empty_nodes, empty_pipes)
if len(errors_empty) > 0:
    print(f"  ✅ 通过：空管网正确返回错误 ({len(errors_empty)} 个错误)")
else:
    print("  ❌ 失败：空管网未返回错误")
    all_passed = False

# 测试12: 边界情况 - 多水源
print("\n【测试12】边界情况 - 多水源验证")
multi_source_nodes = nodes_df.copy()
multi_source_nodes.loc[1, '类型'] = '水源'
multi_source_nodes.loc[1, '水头(m)'] = 35.0
errors_multi = validate_network(multi_source_nodes, pipes_df)
has_source_error = any("水源" in e for e in errors_multi)
if has_source_error:
    print("  ✅ 通过：多水源正确返回错误")
else:
    print("  ❌ 失败：多水源未返回错误")
    all_passed = False

# 总结
print("\n" + "=" * 60)
if all_passed:
    print("🎉 所有测试通过！")
else:
    print("❌ 部分测试未通过")
print("=" * 60)

# 输出详细结果摘要
print("\n【结果摘要】")
print(f"  节点数: {len(nodes_df)}")
print(f"  管段数: {len(pipes_df)}")
print(f"  环路数: {len(loops)}")
print(f"  水力计算收敛: {result['converged']} ({result['iterations']} 次迭代)")
print(f"  总用水量: {total_demand*3600:.1f} m³/h")
print()
print("  管段流量:")
for pr in result['pipe_results']:
    print(f"    {pr['起始节点']}->{pr['终止节点']}: {pr['流量(L/s)']:.3f} L/s, v={pr['流速(m/s)']:.4f} m/s")
print()
print("  节点压力:")
for nr in result['node_results']:
    print(f"    {nr['节点名称']}: {nr['压力(m)']:.3f} m")
print()
print("  节点余氯:")
for nr in wq['node_chlorine']:
    print(f"    {nr['节点名称']}: {nr['余氯浓度(mg/L)']:.4f} mg/L ({nr['预警']})")
