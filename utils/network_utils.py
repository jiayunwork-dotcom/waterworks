import numpy as np
import networkx as nx
from collections import deque


def validate_network(nodes_df, pipes_df):
    errors = []

    if len(nodes_df) > 20:
        errors.append("节点数不能超过20个")
    if len(pipes_df) > 30:
        errors.append("管段数不能超过30条")

    source_nodes = nodes_df[nodes_df['类型'] == '水源']
    if len(source_nodes) == 0:
        errors.append("必须至少有一个水源节点")
    elif len(source_nodes) > 1:
        errors.append("必须有且只有一个水源节点")

    if len(nodes_df) == 0 or len(pipes_df) == 0:
        if not errors:
            errors.append("管网数据为空")
        return errors

    node_names = set(nodes_df['节点名称'].tolist())
    for _, row in pipes_df.iterrows():
        start = row['起始节点']
        end = row['终止节点']
        if start not in node_names:
            errors.append(f"管段起始节点 '{start}' 不存在")
        if end not in node_names:
            errors.append(f"管段终止节点 '{end}' 不存在")
        if start == end:
            errors.append(f"管段起始节点和终止节点不能相同: '{start}'")

    duplicate_pipes = pipes_df.duplicated(subset=['起始节点', '终止节点'], keep=False)
    if duplicate_pipes.any():
        dup = pipes_df[duplicate_pipes]
        for _, row in dup.iterrows():
            errors.append(f"重复管段: {row['起始节点']} → {row['终止节点']}")

    if not errors or (len(errors) == 1 and "水源" in errors[0] and len(source_nodes) > 1):
        G = nx.Graph()
        for _, row in nodes_df.iterrows():
            G.add_node(row['节点名称'])
        for _, row in pipes_df.iterrows():
            if row['起始节点'] in node_names and row['终止节点'] in node_names:
                G.add_edge(row['起始节点'], row['终止节点'])
        if not nx.is_connected(G):
            errors.append("管网必须连通，存在孤立节点")

    return errors


def find_loops(pipes_df):
    """
    识别管网中的独立环路（基环）
    返回: list of lists, 每个子列表是 [(pipe_idx, direction), ...]
          direction=1表示管段方向与环路方向一致，-1表示相反
          环路方向为：沿路径从u到v，再沿弦边从v回到u
    """
    G = nx.Graph()
    pipe_original_dir = {}
    for idx, row in pipes_df.iterrows():
        u, v = row['起始节点'], row['终止节点']
        G.add_edge(u, v, pipe_idx=idx)
        pipe_original_dir[(u, v)] = (idx, 1)
        pipe_original_dir[(v, u)] = (idx, -1)

    if not nx.is_connected(G):
        return []

    # 用生成树法找基环
    spanning_tree = nx.minimum_spanning_tree(G)
    chords = []
    for u, v in G.edges():
        if not spanning_tree.has_edge(u, v):
            chords.append((u, v))

    loops = []
    for u, v in chords:
        try:
            # 在生成树中找u到v的路径
            path = nx.shortest_path(spanning_tree, u, v)
            loop_edges = []

            # 添加树边（沿路径从u到v）
            for i in range(len(path) - 1):
                n1, n2 = path[i], path[i + 1]
                key = (n1, n2)
                if key in pipe_original_dir:
                    pidx, direction = pipe_original_dir[key]
                    loop_edges.append((pidx, direction))

            # 添加弦边（从v回到u，闭合环路）
            key = (v, u)
            if key in pipe_original_dir:
                pidx, direction = pipe_original_dir[key]
                loop_edges.append((pidx, direction))

            loops.append(loop_edges)
        except nx.NetworkXNoPath:
            continue

    return loops


def hazen_williams_head_loss(Q, L, C, D):
    """
    Hazen-Williams公式计算水头损失
    Q: 流量 (m³/s)
    L: 管长 (m)
    C: Hazen-Williams粗糙系数
    D: 管径 (m)
    返回: 水头损失 (m)，符号与流量一致
    """
    if abs(Q) < 1e-15:
        return 0.0
    sign = 1.0 if Q >= 0 else -1.0
    hf = 10.67 * L * (abs(Q) ** 1.852) / ((C ** 1.852) * (D ** 4.87))
    return sign * hf


def hazen_williams_head_loss_derivative(Q, L, C, D):
    """
    水头损失对流量的导数 (用于Hardy-Cross迭代)
    """
    if abs(Q) < 1e-15:
        Q = 1e-15
    return 1.852 * 10.67 * L * (abs(Q) ** 0.852) / ((C ** 1.852) * (D ** 4.87))


def assign_initial_flows(nodes_df, pipes_df):
    """
    分配初始流量，满足节点流量平衡
    方法：构造生成树，从水源向下游分配流量
    """
    source = nodes_df[nodes_df['类型'] == '水源']['节点名称'].iloc[0]
    demands = {}
    for _, row in nodes_df.iterrows():
        if row['节点名称'] != source:
            demands[row['节点名称']] = row.get('用水量(m³/h)', 0) / 3600.0
        else:
            demands[row['节点名称']] = 0.0

    # 构建无向图
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['节点名称'])
    pipe_map = {}
    for idx, row in pipes_df.iterrows():
        u, v = row['起始节点'], row['终止节点']
        G.add_edge(u, v, pipe_idx=idx)
        pipe_map[(u, v)] = idx
        pipe_map[(v, u)] = idx

    # 用BFS构造以水源为根的生成树
    parent = {}
    children = {n: [] for n in G.nodes()}
    visited = {source}
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                queue.append(neighbor)

    # 后序遍历，计算每个子树的总需水量
    downstream_demand = {}
    order = list(nx.bfs_tree(G, source))
    for node in reversed(order):
        total = demands.get(node, 0)
        for child in children.get(node, []):
            total += downstream_demand.get(child, 0)
        downstream_demand[node] = total

    # 分配流量
    flows = np.zeros(len(pipes_df))
    for node in order:
        for child in children.get(node, []):
            # 找到连接node和child的管段
            key = (node, child)
            pidx = pipe_map.get(key)
            if pidx is not None:
                row = pipes_df.loc[pidx]
                if row['起始节点'] == node and row['终止节点'] == child:
                    flows[pidx] = downstream_demand.get(child, 0.001)
                else:
                    flows[pidx] = -downstream_demand.get(child, 0.001)

    # 处理总需求为0的情况
    total_demand = sum(demands.values())
    if total_demand < 1e-12:
        for idx in range(len(pipes_df)):
            flows[idx] = 0.001

    return flows


def hardy_cross_calculation(nodes_df, pipes_df, max_iter=500, tol=0.001):
    """
    Hardy-Cross法求解环状管网流量分配
    
    参数:
        nodes_df: 节点DataFrame
        pipes_df: 管段DataFrame
        max_iter: 最大迭代次数
        tol: 收敛阈值 (m³/s)
    
    返回:
        dict 包含管段结果、节点结果、收敛状态等
    """
    loops = find_loops(pipes_df)

    flows = assign_initial_flows(nodes_df, pipes_df)

    # 预计算管段参数
    pipe_data = []
    for idx, row in pipes_df.iterrows():
        L = row['管长(m)']
        D = row['管径(mm)'] / 1000.0
        C = row.get('粗糙系数', 130)
        pipe_data.append({'L': L, 'D': D, 'C': C})

    converged = False
    iteration = 0

    # Hardy-Cross迭代
    for iteration in range(max_iter):
        max_correction = 0.0

        for loop in loops:
            sum_hf = 0.0      # 环路水头损失之和
            sum_dhf = 0.0     # 水头损失导数绝对值之和

            for pipe_idx, direction in loop:
                Q = flows[pipe_idx] * direction
                pd = pipe_data[pipe_idx]
                hf = hazen_williams_head_loss(Q, pd['L'], pd['C'], pd['D'])
                dhf = hazen_williams_head_loss_derivative(abs(Q), pd['L'], pd['C'], pd['D'])
                sum_hf += hf
                sum_dhf += dhf

            # 计算流量修正量
            if sum_dhf > 1e-15:
                delta_q = -sum_hf / sum_dhf
            else:
                delta_q = 0.0

            max_correction = max(max_correction, abs(delta_q))

            # 修正环路中各管段的流量
            for pipe_idx, direction in loop:
                flows[pipe_idx] += delta_q * direction

        # 检查收敛
        if max_correction < tol:
            converged = True
            break

    # 计算管段结果
    pipe_results = []
    for idx, row in pipes_df.iterrows():
        Q_m3s = flows[idx]
        Q_Ls = Q_m3s * 1000.0
        D = pipe_data[idx]['D']
        area = np.pi * (D ** 2) / 4.0
        velocity = Q_m3s / area if area > 0 else 0
        hf = hazen_williams_head_loss(Q_m3s, pipe_data[idx]['L'], pipe_data[idx]['C'], pipe_data[idx]['D'])
        pipe_results.append({
            'pipe_idx': idx,
            '起始节点': row['起始节点'],
            '终止节点': row['终止节点'],
            '管径(mm)': row['管径(mm)'],
            '管长(m)': row['管长(m)'],
            '粗糙系数': pipe_data[idx]['C'],
            '流量(m³/s)': Q_m3s,
            '流量(L/s)': Q_Ls,
            '流速(m/s)': abs(velocity),
            '水头损失(m)': abs(hf),
        })

    # 计算节点水头（从水源出发，沿流向计算）
    source = nodes_df[nodes_df['类型'] == '水源']['节点名称'].iloc[0]

    # 构建有向图（基于流量方向）
    pipe_upstream = {}
    pipe_downstream = {}
    for idx, row in pipes_df.iterrows():
        Q = flows[idx]
        if Q >= 0:
            pipe_upstream[idx] = row['起始节点']
            pipe_downstream[idx] = row['终止节点']
        else:
            pipe_upstream[idx] = row['终止节点']
            pipe_downstream[idx] = row['起始节点']

    # 从水源开始计算各节点水头
    node_heads = {}
    for _, row in nodes_df.iterrows():
        if row['类型'] == '水源':
            node_heads[row['节点名称']] = row.get('水头(m)', 30.0)
        else:
            node_heads[row['节点名称']] = None

    # 用迭代法计算节点水头（因为可能有环）
    for _ in range(len(nodes_df) + 10):
        changed = False
        for idx in range(len(pipes_df)):
            up = pipe_upstream[idx]
            ds = pipe_downstream[idx]
            hf = pipe_results[idx]['水头损失(m)']

            if node_heads.get(up) is not None:
                new_head = node_heads[up] - hf
                if node_heads.get(ds) is None or abs(node_heads[ds] - new_head) > 1e-8:
                    node_heads[ds] = new_head
                    changed = True
        if not changed:
            break

    # 计算节点结果
    node_results = []
    for _, row in nodes_df.iterrows():
        name = row['节点名称']
        head = node_heads.get(name, 0)
        elevation = row.get('地面标高(m)', 0)
        pressure = head - elevation
        node_results.append({
            '节点名称': name,
            '类型': row['类型'],
            '水头(m)': head,
            '地面标高(m)': elevation,
            '压力(m)': pressure,
            '用水量(m³/h)': row.get('用水量(m³/h)', 0),
        })

    return {
        'pipe_results': pipe_results,
        'node_results': node_results,
        'converged': converged,
        'iterations': iteration + 1,
        'flows': flows,
        'pipe_data': pipe_data,
    }


def calculate_water_quality(nodes_df, pipes_df, flow_results, k_decay=0.5, source_chlorine=1.0):
    """
    计算管网余氯衰减
    
    参数:
        nodes_df: 节点DataFrame
        pipes_df: 管段DataFrame
        flow_results: 水力计算结果
        k_decay: 衰减常数 (1/h)
        source_chlorine: 水源初始余氯浓度 (mg/L)
    
    返回:
        dict 包含节点余氯和管段余氯结果
    """
    source = nodes_df[nodes_df['类型'] == '水源']['节点名称'].iloc[0]
    flows = flow_results['flows']
    pipe_data = flow_results['pipe_data']

    # 计算管段停留时间和流向
    pipe_travel_time = {}
    pipe_upstream = {}
    pipe_downstream = {}

    for idx in range(len(pipes_df)):
        Q = flows[idx]
        D = pipe_data[idx]['D']
        L = pipe_data[idx]['L']
        area = np.pi * (D ** 2) / 4.0
        volume = area * L

        if abs(Q) > 1e-15:
            travel_time_h = (volume / abs(Q)) / 3600.0
        else:
            travel_time_h = 9999.0

        pipe_travel_time[idx] = travel_time_h

        # 判断流向
        if Q >= 0:
            pipe_upstream[idx] = pipes_df.loc[idx, '起始节点']
            pipe_downstream[idx] = pipes_df.loc[idx, '终止节点']
        else:
            pipe_upstream[idx] = pipes_df.loc[idx, '终止节点']
            pipe_downstream[idx] = pipes_df.loc[idx, '起始节点']

    # 构建节点入流管段映射
    node_incoming = {}
    for _, row in nodes_df.iterrows():
        name = row['节点名称']
        node_incoming[name] = []
    for idx in range(len(pipes_df)):
        ds = pipe_downstream[idx]
        node_incoming[ds].append(idx)

    # 初始化节点余氯浓度
    node_chlorine = {}
    for _, row in nodes_df.iterrows():
        name = row['节点名称']
        if name == source:
            node_chlorine[name] = source_chlorine
        else:
            node_chlorine[name] = 0.0

    # 迭代计算（处理环状管网）
    for _ in range(len(nodes_df) * 3 + 50):
        max_change = 0.0
        for _, row in nodes_df.iterrows():
            name = row['节点名称']
            if name == source:
                continue

            incoming_pipes = node_incoming[name]
            if not incoming_pipes:
                continue

            flow_weighted_conc = 0.0
            total_flow = 0.0

            for pidx in incoming_pipes:
                up_node = pipe_upstream[pidx]
                c_up = node_chlorine.get(up_node, 0)
                t = pipe_travel_time[pidx]
                c_after = c_up * np.exp(-k_decay * t)
                Q = abs(flows[pidx])
                flow_weighted_conc += c_after * Q
                total_flow += Q

            if total_flow > 1e-15:
                new_val = flow_weighted_conc / total_flow
            else:
                new_val = 0.0

            change = abs(new_val - node_chlorine[name])
            max_change = max(max_change, change)
            node_chlorine[name] = new_val

        if max_change < 1e-10:
            break

    # 计算管段余氯结果
    pipe_chlorine = []
    for idx, row in pipes_df.iterrows():
        u = pipe_upstream.get(idx, row['起始节点'])
        t = pipe_travel_time[idx]
        c_up = node_chlorine.get(u, 0)
        c_down = c_up * np.exp(-k_decay * t)
        pipe_chlorine.append({
            'pipe_idx': idx,
            '起始节点': row['起始节点'],
            '终止节点': row['终止节点'],
            '上游余氯(mg/L)': round(c_up, 6),
            '下游余氯(mg/L)': round(c_down, 6),
            '停留时间(h)': round(t, 6),
        })

    # 计算节点余氯结果
    node_chlorine_results = []
    for _, row in nodes_df.iterrows():
        name = row['节点名称']
        cl = node_chlorine.get(name, 0)
        warning = cl < 0.05
        node_chlorine_results.append({
            '节点名称': name,
            '类型': row['类型'],
            '余氯浓度(mg/L)': round(cl, 6),
            '预警': '⚠️ 低于0.05' if warning else '正常',
        })

    return {
        'node_chlorine': node_chlorine_results,
        'pipe_chlorine': pipe_chlorine,
        'k_decay': k_decay,
        'source_chlorine': source_chlorine,
    }


def sensitivity_analysis(nodes_df, pipes_df, param_type, param_id, param_range,
                         target_type, target_id, max_iter=500, tol=0.001,
                         k_decay=None, source_chlorine=1.0):
    """
    敏感性分析
    
    参数:
        nodes_df: 节点DataFrame
        pipes_df: 管段DataFrame
        param_type: 参数类型 ('管径', '粗糙系数', '用水量')
        param_id: 参数标识 (管段索引或节点名称)
        param_range: 参数值范围数组
        target_type: 目标指标类型 ('压力', '余氯')
        target_id: 目标节点名称
        max_iter: 最大迭代次数
        tol: 收敛阈值
        k_decay: 衰减常数 (仅用于余氯计算)
        source_chlorine: 水源余氯浓度
    
    返回:
        list of dict, 每个包含 '参数值' 和 '目标值'
    """
    results = []

    if k_decay is None:
        k_decay = 0.5

    for param_value in param_range:
        nodes_copy = nodes_df.copy()
        pipes_copy = pipes_df.copy()

        if param_type == '管径':
            mask = pipes_copy.index == param_id
            pipes_copy.loc[mask, '管径(mm)'] = param_value
        elif param_type == '粗糙系数':
            mask = pipes_copy.index == param_id
            pipes_copy.loc[mask, '粗糙系数'] = param_value
        elif param_type == '用水量':
            mask = nodes_copy['节点名称'] == param_id
            nodes_copy.loc[mask, '用水量(m³/h)'] = param_value

        hc_result = hardy_cross_calculation(nodes_copy, pipes_copy, max_iter, tol)

        if target_type == '压力':
            for nr in hc_result['node_results']:
                if nr['节点名称'] == target_id:
                    results.append({
                        '参数值': param_value,
                        '目标值': nr['压力(m)'],
                    })
                    break
        elif target_type == '余氯':
            wq_result = calculate_water_quality(nodes_copy, pipes_copy, hc_result, k_decay, source_chlorine)
            for nr in wq_result['node_chlorine']:
                if nr['节点名称'] == target_id:
                    results.append({
                        '参数值': param_value,
                        '目标值': nr['余氯浓度(mg/L)'],
                    })
                    break

    return results


def compute_network_layout(nodes_df, pipes_df):
    """
    计算管网布局坐标
    """
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['节点名称'])
    for _, row in pipes_df.iterrows():
        G.add_edge(row['起始节点'], row['终止节点'])

    try:
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)
    except Exception:
        pos = {}
        for i, node in enumerate(G.nodes()):
            pos[node] = (i % 5, i // 5)

    return pos


CANDIDATE_DIAMETERS = [100, 150, 200, 250, 300, 350, 400, 450, 500]


def calculate_pipe_cost(diameter_mm, length_m):
    """
    计算单条管段的管材费用
    单价 = 0.5 * 管径(mm) 元/m
    """
    unit_price = 0.5 * diameter_mm
    return unit_price * length_m


def calculate_total_cost(pipes_df):
    """
    计算管网管材总费用
    """
    total_cost = 0.0
    for _, row in pipes_df.iterrows():
        total_cost += calculate_pipe_cost(row['管径(mm)'], row['管长(m)'])
    return total_cost


def decode_chromosome(chromosome, pipes_df):
    """
    将染色体（管径索引数组）解码为管段DataFrame
    """
    pipes_copy = pipes_df.copy()
    for i, idx in enumerate(chromosome):
        pipes_copy.loc[i, '管径(mm)'] = CANDIDATE_DIAMETERS[idx]
    return pipes_copy


def evaluate_fitness(chromosome, nodes_df, pipes_df, min_pressure, max_iter=500, tol=0.001):
    """
    计算个体适应度
    满足约束: 适应度 = 1 / 总费用
    不满足约束: 适应度 = 0
    """
    pipes_decoded = decode_chromosome(chromosome, pipes_df)
    result = hardy_cross_calculation(nodes_df, pipes_decoded, max_iter=max_iter, tol=tol)

    if not result['converged']:
        return 0.0, float('inf')

    for nr in result['node_results']:
        if nr['类型'] == '需求' and nr['压力(m)'] < min_pressure:
            return 0.0, float('inf')

    total_cost = calculate_total_cost(pipes_decoded)
    fitness = 1.0 / total_cost if total_cost > 0 else 0.0
    return fitness, total_cost


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    锦标赛选择
    """
    n = len(population)
    selected = []
    for _ in range(n):
        candidates_idx = np.random.choice(n, tournament_size, replace=False)
        best_idx = candidates_idx[0]
        for idx in candidates_idx:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
        selected.append(population[best_idx].copy())
    return selected


def single_point_crossover(parent1, parent2, crossover_rate):
    """
    单点交叉
    """
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    if n <= 1:
        return parent1.copy(), parent2.copy()

    point = np.random.randint(1, n)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def mutate(chromosome, mutation_rate):
    """
    变异：随机将某个基因位替换为候选管径列表中的随机值
    """
    mutated = chromosome.copy()
    n = len(mutated)
    n_choices = len(CANDIDATE_DIAMETERS)

    for i in range(n):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.randint(0, n_choices)

    return mutated


def initialize_population(pop_size, n_pipes):
    """
    初始化种群
    """
    n_choices = len(CANDIDATE_DIAMETERS)
    population = []
    for _ in range(pop_size):
        chromosome = np.random.randint(0, n_choices, size=n_pipes)
        population.append(chromosome)
    return population


def genetic_algorithm_optimization(nodes_df, pipes_df, min_pressure=15.0,
                                    pop_size=50, max_generations=100,
                                    crossover_rate=0.8, mutation_rate=0.1,
                                    tournament_size=3, max_iter_hc=500, tol_hc=0.001,
                                    progress_callback=None):
    """
    遗传算法求解管径优化问题

    参数:
        nodes_df: 节点DataFrame
        pipes_df: 管段DataFrame
        min_pressure: 最低压力阈值 (m)
        pop_size: 种群大小
        max_generations: 最大迭代代数
        crossover_rate: 交叉率
        mutation_rate: 变异率
        tournament_size: 锦标赛选择规模
        max_iter_hc: Hardy-Cross最大迭代次数
        tol_hc: Hardy-Cross收敛阈值
        progress_callback: 进度回调函数 callback(generation, best_cost)

    返回:
        dict 包含优化结果
    """
    n_pipes = len(pipes_df)
    n_choices = len(CANDIDATE_DIAMETERS)

    population = initialize_population(pop_size, n_pipes)

    best_fitness_history = []
    best_cost_history = []
    best_solution = None
    best_cost = float('inf')
    best_fitness = 0.0

    for gen in range(max_generations):
        fitnesses = []
        costs = []
        for chromosome in population:
            fitness, cost = evaluate_fitness(
                chromosome, nodes_df, pipes_df, min_pressure,
                max_iter=max_iter_hc, tol=tol_hc
            )
            fitnesses.append(fitness)
            costs.append(cost)

        fitnesses = np.array(fitnesses)
        costs = np.array(costs)

        best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[best_idx]
        current_best_cost = costs[best_idx]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_cost = current_best_cost
            best_solution = population[best_idx].copy()

        best_fitness_history.append(best_fitness)
        best_cost_history.append(best_cost)

        if progress_callback is not None:
            progress_callback(gen + 1, best_cost)

        if np.all(fitnesses == 0):
            new_population = initialize_population(pop_size, n_pipes)
        else:
            selected = tournament_selection(population, fitnesses, tournament_size)

            new_population = []
            for i in range(0, pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, pop_size - 1)]
                child1, child2 = single_point_crossover(parent1, parent2, crossover_rate)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)

            if len(new_population) > pop_size:
                new_population = new_population[:pop_size]

        population = new_population

    if best_solution is not None:
        best_pipes = decode_chromosome(best_solution, pipes_df)
        best_result = hardy_cross_calculation(nodes_df, best_pipes, max_iter=max_iter_hc, tol=tol_hc)
    else:
        best_pipes = pipes_df.copy()
        best_result = None

    return {
        'best_pipes': best_pipes,
        'best_cost': best_cost,
        'best_fitness': best_fitness,
        'best_hydraulic_result': best_result,
        'cost_history': best_cost_history,
        'fitness_history': best_fitness_history,
        'generations': max_generations,
        'pop_size': pop_size,
        'crossover_rate': crossover_rate,
        'mutation_rate': mutation_rate,
        'min_pressure': min_pressure,
    }
