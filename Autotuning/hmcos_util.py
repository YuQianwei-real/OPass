from time import time
from networkx import DiGraph
from typing import List, cast, Tuple, Optional, Dict
from .graph.abs import GraphAbsForMem
from Autotuning.serenity_eval import _preprocess, _del_lets, _zero_indegree, _zero_outdegree, _alloc_memory, _predecessors

def is_valid_sequence(sequence, G:DiGraph) -> bool:
    """
    判断一个节点序列是否满足 HMCOS 中 Sequence 的条件。
    
    参数:
    - sequence: 节点列表，表示待判断的序列。
    - G: node_graph
    
    返回:
    - bool: 是否为有效的 Sequence。
    """
    # 如果序列长度为 0 或 1，直接视为有效
    if len(sequence) <= 1:
        return True

    # 1. 图结构条件检查（线性链式）
    for i, node in enumerate(sequence):
        if i == 0:
            # 首节点：后继只有一个且必须为下一个节点
            suc_imm = [u for u in G.successors(node) if G.nodes[u]['type'] == 'tensor'] #[]*1
            suc_op = []
            for v in suc_imm:
                suc_op.append([u for u in G.successors(v) if G.nodes[u]['type'] == 'op']) #[[]*n]
            #print('suc_op=',suc_op)
            if len(suc_op[0]) != 1 or suc_op[0][0] != sequence[i+1]:
                return False
        elif i == len(sequence) - 1:
            # 尾节点：前驱只有一个且必须为前一个节点
            pre_imm = [u for u in G.predecessors(node) if G.nodes[u]['type'] == 'tensor'] #[]*n
            pre_op = [] #[[]*1]*n
            for v in pre_imm:
                pre_op.append([u for u in G.predecessors(v) if G.nodes[u]['type'] == 'op'])

            if len(pre_imm) != 1 or pre_op[0][0] != sequence[i-1]:
                return False
        else:
            # 中间节点：前驱和后继必须各为一个，且连接正确
            pre_imm = [u for u in G.predecessors(node) if G.nodes[u]['type'] == 'tensor']
            pre_op = []
            for v in pre_imm:
                pre_op.append([u for u in G.predecessors(v) if G.nodes[u]['type'] == 'op'])
            suc_imm = [u for u in G.successors(node) if G.nodes[u]['type'] == 'tensor']
            suc_op = []
            for v in suc_imm:
                suc_op.append([u for u in G.successors(v) if G.nodes[u]['type'] == 'op'])

            if len(pre_imm) != 1 or pre_op[0][0] != sequence[i-1]:
                return False
            if len(suc_op[0]) != 1 or suc_op[0][0] != sequence[i+1]:
                return False
    """
    # 2. 内存状态条件检查（可选）
    
        #raise ValueError("内存状态列表长度需与序列一致")

        # 条件1：首节点的稳定内存峰值最高
        max_stable = max(stable_footprints)
        if stable_footprints[0] < max_stable:
            return False

        # 条件2：瞬态内存非递增
        for i in range(1, len(transient_footprints)):
            if transient_footprints[i] > transient_footprints[i-1]:
                return False
    """
    return True



def compute_op_lifetimes(G: DiGraph, schedule: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    计算每个操作符的生命周期(alloc_time, free_time)
    
    :param G: NetworkX有向图,节点包含'type'属性('op'或'tensor')
    :param schedule: 操作符调度顺序列表(如['op1', 'op2', ...])
    :return: 字典，键为操作符名称，值为(alloc_time, free_time)
    """
    # 记录操作符的调度时间
    op_time = {op: idx + 1 for idx, op in enumerate(schedule)}
    lifetimes = {}

    for op in schedule:
        assert G.nodes[op]['type'] == 'op'

        alloc = op_time[op]
        free = alloc  # 默认释放时间为自身调度时间（无输出或输出未被使用）
        
        # 获取操作符的输出张量
        outputs = [v for v in G.successors(op) if G.nodes[v]['type'] == 'tensor']
        if not outputs:
            lifetimes[op] = (alloc, alloc)
            continue
        
        # 计算所有输出张量的最晚释放时间
        max_free = 0
        for tensor in outputs:
            # 找到使用该张量的所有后继操作符
            users = [u for u in G.successors(tensor) if G.nodes[u]['type'] == 'op']
            if not users:
                # 输出张量未被使用（如网络输出），释放时间为调度结束时间
                tensor_free = len(schedule)
            else:
                # 释放时间为最后一个使用该张量的操作符的调度时间
                tensor_free = max(op_time[user] for user in users)
            max_free = max(max_free, tensor_free)
        
        lifetimes[op] = (alloc, max_free)
    
    return lifetimes

def estimate_peak(G: DiGraph, seq: List[str]) -> float:
    
    alloc = {}  # 张量分配时间
    free = {}    # 张量释放时间
    scheduled_ops = set(seq)
    inputs = []  #input nodes

    ignore_nodes = [] #nodes to ignore
                
    for n, ndata in G.nodes.items():
        if cast(str, n).startswith('in'):
            inputs.append(n)
        
        if ndata['type'] != 'op':
            continue

        if ndata['op'] in ('let', 'tuple', 'getitem'):
            for tn in G.successors(n):
                assert G.nodes[tn]['type'] == 'tensor'
                if tn not in ignore_nodes:
                    ignore_nodes.append(tn)
            continue
    
    # 处理输入张量
    for tensor in inputs:
        alloc[tensor] = 0
        users = [op for op in G.successors(tensor) 
            if G.nodes[op]['type'] == 'op' and op in scheduled_ops]
        if users:
            free[tensor] = max(seq.index(u) + 1 for u in users)
        else:
            free[tensor] = len(seq)
    
    # 模拟调度过程
    for time, op in enumerate(seq, start=1):
        # 处理输出张量
        for tensor in G.successors(op):
            if G.nodes[tensor]['type'] == 'tensor':
                alloc[tensor] = time
                users = [u for u in G.successors(tensor) if G.nodes[u]['type'] == 'op' and u in scheduled_ops]
                if not users:
                    free[tensor] = len(seq)  # 输出张量未被使用
                else:
                    free[tensor] = max(seq.index(u) + 1 for u in users)
        
        # 处理输入张量
        for tensor in G.predecessors(op):
            if G.nodes[tensor]['type'] == 'tensor':
                # 检查是否还有后续使用
                remaining_users = [u for u in G.successors(tensor) if u in seq[time:] and G.nodes[u]['type'] == 'op']
                if not remaining_users:
                    free[tensor] = time
    
    # 计算每个时间点的内存占用
    peak = 0.0
    for t in range(1, len(seq)+1):
        current = 0.0
        for tensor in G.nodes:
            if G.nodes[tensor]['type'] == 'tensor' and tensor not in ignore_nodes:
                a = alloc.get(tensor, float('inf'))
                f = free.get(tensor, 0)
                if a <= t <= f:
                    current += G.nodes[tensor]['mem'] / (8 * 1024 * 1024)
        peak = max(peak, current)
    
    return peak