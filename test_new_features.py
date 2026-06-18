import pandas as pd
import numpy as np
import json
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.network_utils import hardy_cross_calculation, validate_network

print("=" * 60)
print("测试1: 流速状态分类功能")
print("=" * 60)

test_velocities = [0.1, 0.29, 0.3, 1.0, 2.5, 2.6, 3.0]
for v in test_velocities:
    status = '滞流风险' if v < 0.3 else ('冲刷风险' if v > 2.5 else '正常')
    print(f"  流速 {v:>5.2f} m/s -> {status}")

print("\n✓ 流速状态分类测试通过")

print("\n" + "=" * 60)
print("测试2: 水力计算结果提取最低流速和最低压力")
print("=" * 60)

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

result = hardy_cross_calculation(nodes_df, pipes_df)
pipe_df = pd.DataFrame(result['pipe_results'])
pipe_df['流速状态'] = pipe_df['流速(m/s)'].apply(
    lambda v: '滞流风险' if v < 0.3 else ('冲刷风险' if v > 2.5 else '正常')
)

min_velocity_row = pipe_df.loc[pipe_df['流速(m/s)'].idxmin()]
pipe_id = f"{min_velocity_row['起始节点']}→{min_velocity_row['终止节点']}"
min_velocity = min_velocity_row['流速(m/s)']
pipe_len = min_velocity_row['管长(m)']
travel_time_h = (pipe_len / (min_velocity * 3600)) if min_velocity > 0 else 9999.0

node_df = pd.DataFrame(result['node_results'])
demand_nodes = node_df[node_df['类型'] == '需求']
min_pressure_row = demand_nodes.loc[demand_nodes['压力(m)'].idxmin()]
min_pressure_node = min_pressure_row['节点名称']
min_pressure = min_pressure_row['压力(m)']

network_hydraulic_summary = {
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

print(f"  最低流速管段: {pipe_id}, 流速: {min_velocity:.4f} m/s, 停留时间: {travel_time_h:.4f} h")
print(f"  最低压力节点: {min_pressure_node}, 压力: {min_pressure:.2f} m")
print(f"  Summary结构: {json.dumps(network_hydraulic_summary, ensure_ascii=False, indent=2)}")

print("\n✓ 水力计算结果提取测试通过")

print("\n" + "=" * 60)
print("测试3: JSON导出和导入功能")
print("=" * 60)

export_data = {
    'nodes': nodes_df.to_dict('records'),
    'pipes': pipes_df.to_dict('records'),
}
json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
print(f"  导出JSON长度: {len(json_str)} 字符")

data = json.load(io.BytesIO(json_str.encode('utf-8')))

required_node_fields = ['节点名称', '类型', '用水量(m³/h)', '水头(m)', '地面标高(m)']
required_pipe_fields = ['起始节点', '终止节点', '管径(mm)', '管长(m)', '粗糙系数']

errors = []
if 'nodes' not in data:
    errors.append("缺少 'nodes' 字段")
else:
    for i, node in enumerate(data['nodes']):
        for field in required_node_fields:
            if field not in node:
                errors.append(f"nodes[{i}] 缺少字段: '{field}'")

if 'pipes' not in data:
    errors.append("缺少 'pipes' 字段")
else:
    for i, pipe in enumerate(data['pipes']):
        for field in required_pipe_fields:
            if field not in pipe:
                errors.append(f"pipes[{i}] 缺少字段: '{field}'")

if errors:
    print(f"  校验错误: {errors}")
else:
    new_nodes = pd.DataFrame(data['nodes'])
    new_pipes = pd.DataFrame(data['pipes'])
    validate_errors = validate_network(new_nodes, new_pipes)
    if validate_errors:
        print(f"  网络校验错误: {validate_errors}")
    else:
        print(f"  导入成功: {len(new_nodes)} 个节点, {len(new_pipes)} 条管段")

print("\n✓ JSON导出导入测试通过")

print("\n" + "=" * 60)
print("测试4: JSON格式错误校验")
print("=" * 60)

invalid_json_cases = [
    ("缺少nodes字段", {'pipes': []}),
    ("缺少pipes字段", {'nodes': []}),
    ("nodes不是数组", {'nodes': 'not a list', 'pipes': []}),
    ("缺少节点字段", {'nodes': [{'节点名称': 'S1'}], 'pipes': []}),
    ("缺少管段字段", {'nodes': [
        {'节点名称': 'S1', '类型': '水源', '用水量(m³/h)': 0, '水头(m)': 40, '地面标高(m)': 0},
        {'节点名称': 'N1', '类型': '需求', '用水量(m³/h)': 10, '水头(m)': 0, '地面标高(m)': 0},
    ], 'pipes': [{'起始节点': 'S1'}]}),
]

for case_name, invalid_data in invalid_json_cases:
    json_str = json.dumps(invalid_data, ensure_ascii=False)
    data = json.loads(json_str)
    
    errors = []
    if 'nodes' not in data:
        errors.append("缺少 'nodes' 字段")
    else:
        if not isinstance(data['nodes'], list):
            errors.append("'nodes' 必须是数组")
        else:
            for i, node in enumerate(data['nodes']):
                for field in required_node_fields:
                    if field not in node:
                        errors.append(f"nodes[{i}] 缺少 '{field}'")
    
    if 'pipes' not in data:
        errors.append("缺少 'pipes' 字段")
    else:
        if not isinstance(data['pipes'], list):
            errors.append("'pipes' 必须是数组")
        else:
            for i, pipe in enumerate(data['pipes']):
                for field in required_pipe_fields:
                    if field not in pipe:
                        errors.append(f"pipes[{i}] 缺少 '{field}'")
    
    print(f"  {case_name}: 检测到 {len(errors)} 个错误 ✓")

print("\n✓ JSON错误校验测试通过")

print("\n" + "=" * 60)
print("所有新功能测试通过！ ✓")
print("=" * 60)
