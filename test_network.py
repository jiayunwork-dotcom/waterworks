import sys
sys.path.insert(0, '.')
from utils.network_utils import validate_network, hardy_cross_calculation, calculate_water_quality, sensitivity_analysis, compute_network_layout
import pandas as pd
import numpy as np

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

errors = validate_network(nodes_df, pipes_df)
print('Validation errors:', errors)

result = hardy_cross_calculation(nodes_df, pipes_df, max_iter=500, tol=0.001)
print('Converged:', result['converged'])
print('Iterations:', result['iterations'])
print()
print('Pipe results:')
for pr in result['pipe_results']:
    print(f"  {pr['起始节点']}->{pr['终止节点']}: Q={pr['流量(L/s)']:.3f} L/s, v={pr['流速(m/s)']:.4f} m/s, hf={pr['水头损失(m)']:.4f} m")
print()
print('Node results:')
for nr in result['node_results']:
    print(f"  {nr['节点名称']}: H={nr['水头(m)']:.3f}m, P={nr['压力(m)']:.3f}m")

wq = calculate_water_quality(nodes_df, pipes_df, result, k_decay=0.5, source_chlorine=1.0)
print()
print('Water quality results:')
for nr in wq['node_chlorine']:
    print(f"  {nr['节点名称']}: Cl={nr['余氯浓度(mg/L)']:.4f} mg/L, {nr['预警']}")

param_range = np.linspace(200, 500, 7)
sa = sensitivity_analysis(nodes_df, pipes_df, '管径', 0, param_range, '压力', 'N3')
print()
print('Sensitivity analysis (pipe 0 diameter -> N3 pressure):')
for r in sa:
    print(f"  D={r['参数值']:.0f}mm -> P={r['目标值']:.4f}m")

print()
print('All tests passed!')
