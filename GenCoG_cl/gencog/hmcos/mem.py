from GenCoG_cl.gencog.graph.base import Operation, Graph, Value, Input, Output
from GenCoG_cl.gencog.graph.viz import visualize
from GenCoG_cl.gencog.graph.relay import build_graph
from GenCoG_cl.gencog.solve.solver import TensorType

from typing import List, cast, Tuple, Optional, Dict
from tvm import relay
from networkx import DiGraph


ewOps = [
    "abs",        "add",   "and",   "neg",         "mul",
    "exp",        "div",   "ceil",  "not",         "leakyrelu",
    "elu",        "equal", "floor", "greater",     "hardsigmoid",
    "selu",       "less",  "prelu", "log",         "or",
    "reciprocal", "pow",   "relu",  "sigmoid",     "softplus",
    "softsign",   "sqrt",  "sub",   "tanh",        "xor",
    "acos",       "asin",  "atan",  "cos",         "sin",
    "tan",        "sinh",  "cosh",  "asinh",       "acosh",
    "atanh",      "sign",  "erf",   "mod",         "thresholdedrelu",
    "bitshift",   "round", "celu",  "lessorequal", "greaterorequal",
    "hardswish",  "clip"]

reinterpOps = ["squeeze", "unsqueeze", "reshape", "flatten"]

def estimate_peak_hmcos0(op_seq: List[Operation], G: Graph) -> float:
    alloc = {}  # 张量分配时间
    free = {}    # 张量释放时间
    scheduled_ops = set(op_seq)
    input_vals = []  #input nodes
    ignore_vals = [] 
    graph_vals = [] # all value nodes
    peakVals = [] #peak values
    
    for in_ in G.inputs_:
        input_vals.append(in_.value_)
    for con_ in G.constants_:
        input_vals.append(con_.value_)
    
    for op in G.oprs_:
        if op.op_.name_ in ('let', 'tuple', 'getitem'):
            assert len(op.outputs_) == 1
            ignore_vals.append(op.outputs_[0])
    # 处理输入张量
    for in_val in input_vals:
        if in_val not in graph_vals:
            graph_vals.append(in_val)
        alloc[in_val] = 0
        users = [op for op in in_val.uses_ 
                 if isinstance(op, Operation) and op in scheduled_ops]
        if users: 
            free[in_val] = max(op_seq.index(u) + 1 for u in users)
        else:
            free[in_val] = len(op_seq)
    # 模拟调度过程
    for time, op in enumerate(op_seq, start=1):
        # 处理输出张量
        for out_val in op.outputs_:
            assert isinstance(out_val, Value)
            if out_val not in graph_vals:
                graph_vals.append(in_val)
            alloc[out_val] = time
            users = [u for u in out_val.uses_ if isinstance(u, Operation) and u in scheduled_ops]
            if not users:
                free[out_val] = len(op_seq)  # 输出张量未被使用
            else:
                free[out_val] = max(op_seq.index(u) + 1 for u in users)
        
        # 处理输入张量
        for in_val in op.inputs_:
            assert isinstance(in_val, Value)
            if in_val not in graph_vals:
                graph_vals.append(in_val)
            # 检查是否还有后续使用
            remaining_users = [u for u in in_val.uses_ if u in op_seq[time:] and isinstance(u, Operation)]
            if not remaining_users:
                free[in_val] = time
    
    # 计算每个时间点的内存占用
    peak = 0.0
    for t in range(1, len(op_seq)+1):
        current = 0.0
        currentVals: List[Value] = [] 
        for v in graph_vals:
            if isinstance(v, Value) and v not in ignore_vals:
                a = alloc.get(v, float('inf'))
                f = free.get(v, 0)
                if a <= t <= f:
                    current += v.type_.memo_bytes / (1024 * 1024)
                    currentVals.append(v)
                    if isinstance(v.def_, Operation):
                        idx = overlap_input(v.def_) 
                        for j, input in enumerate(v.def_.inputs_):
                            if idx == j:  # 可重叠，立即释放
                                current -= v.type_.memo_bytes/ (1024 * 1024)
        if current > peak:
            peak = current
            peakVals = currentVals
        #print(current)
    
    return peak, peakVals

def estimate_peak_networkx(seq: List[str], G: DiGraph) -> float:
    
    alloc = {}  # 张量分配时间
    free = {}    # 张量释放时间
    scheduled_ops = set(seq)
    inputs = []  #input nodes

    ignore_nodes = [] #nodes to ignore
                
    for n, ndata in G.nodes.items():
        if cast(str, n).startswith('in') or cast(str, n).startswith('con'):
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

#目前还存在问题
def preprocess_op_seq(G: Graph) -> Graph: 
    """
    预处理 Relay 图，移除 let/getitem/tuple 节点，并将它们的输入值
    直接传递给后续节点。
    """
    replace_map = {}  # 输出 Value -> 输入 Value 的映射
    op_seq = G.oprs_

    # Step 1: 构建 replace_map（忽略 let/getitem/tuple 节点）
    for op in op_seq:
        if op.op_.name_ in ['let', 'getitem', 'tuple']:
            assert len(op.outputs_) == 1
            # 每个输 Value 对应一个输 Value
            
            replace_map[op.outputs_[0]] = op.inputs_

    # Step 2: 替换输入值并更新 uses_ 列表
    new_op_seq = []
    for op in op_seq:
        #if op.op_.name_ in ['let', 'getitem', 'tuple']:
        #    continue  # 跳过这些节点

        # 替换输入值
        for i in range(len(op.inputs_)):
            val = op.inputs_[i]
            if val in replace_map:
                old_val = val
                new_val_list = replace_map[old_val]

                # 更新 uses_ 列表
                if op in old_val.uses_:
                    old_val.uses_.remove(op)
                for new_val in new_val_list:
                    new_val.uses_.append(op)
                    op.inputs_.append(new_val)

                op.inputs_.remove(old_val)

        if op.op_.name_ not in ['let', 'getitem', 'tuple']:
            new_op_seq.append(op)

    G.oprs_ = new_op_seq
    return G


def estimate_peak_hmcos(op_seq: List[Operation], G: Graph):
    # 初始化总内存和使用计数
    total: int = 0
    use_cnt: Dict[Value: int] = {}
    next_kill: List[Value] = []  # 下一时间步释放的值
    
    # 初始化输入的内存占用
    for in_vert in G.inputs_:
        in_val = in_vert.value_
        use_cnt[in_val] = len(in_val.uses_)
        for op in in_val.uses_:
            if isinstance(op, Operation) and op.op_.name_ in ('let','getitem','tuple'):
                use_cnt[in_val] += len(op.outputs_[0].uses_) -1
        total += in_val.type_.memo_bytes  # return number of bytes
    for con_vert in G.constants_:
        con_val = con_vert.value_
        use_cnt[con_val] = len(con_val.uses_)
        for op in con_val.uses_:
            if isinstance(op, Operation) and op.op_.name_ in ('let','getitem','tuple'):
                use_cnt[con_val] += len(op.outputs_[0].uses_) -1
        total += con_val.type_.memo_bytes

    peak = total  # 初始峰值内存

    # 遍历操作序列
    for i, op in enumerate(op_seq):
        if op.op_.name_ in ('let', 'tuple', 'getitem'):
            continue

        use_op_cnt = []
        next_kill_op: List[str] = []
        #print(i, op.op_.name_)
        # 生成输出：增加内存占用
        for out in op.outputs_:
            use_cnt[out] = len(out.uses_)
            for use_op in out.uses_:
                if isinstance(use_op, Operation) and use_op.op_.name_ in ('let','getitem','tuple'):
                    use_cnt[out] += len(use_op.outputs_[0].uses_) -1
            total += out.type_.memo_bytes

        # 释放上一时间步标记的值
        for val in next_kill:
            total -= val.type_.memo_bytes
        next_kill.clear()


        # 处理输入：更新使用计数并可能释放内存
        ovl_idx = overlap_input(op)  # 需实现 overlap_input 函数
        for j, in_ in enumerate(op.inputs_):
            if in_.type_.memo_bytes <5:
                continue
            if isinstance(in_.def_, Operation) and in_.def_.op_.name_ in ('let', 'tuple', 'getitem'):
                ignore_op = in_.def_
                for in_before in ignore_op.inputs_:
                    # 判断是否释放内存
                    use_cnt[in_before] -= 1 

                    if use_cnt[in_before] == 0:
                        next_kill.append(in_before)
                        del use_cnt[in_before]
            elif in_ not in use_cnt:    
                use_cnt[in_] = 1
            #    raise ValueError(f"Value {in_.type_} used without definition by Operation {op.op_.name_} before.") 
            else:
                use_cnt[in_] -= 1

            # 判断是否释放内存
                if use_cnt[in_] == 0:
                    if ovl_idx == j:  # 可重叠，立即释放
                        total -= in_.type_.memo_bytes
                    else:  # 延迟释放到下一时间步
                #    next_kill.append(in_)
                        next_kill.append(in_)
                    del use_cnt[in_]

        # 更新峰值内存
        peak = max(peak, total)
        #print('\tuse_cnt = ',use_cnt)
        for in_, num in use_cnt.items():
            if isinstance(in_.def_, Operation):
                use_op_cnt.append([in_.def_.op_.name_, [num, in_.type_]])
            else:
                use_op_cnt.append([in_.def_, num])
        #print('\tuse_op_cnt = ',use_op_cnt)
        #print('\tnext_kill = ',next_kill)
        for val_ in next_kill:
            if isinstance(val_.def_, Operation):
                next_kill_op.append(val_.def_.op_.name_)
            else:
                next_kill_op.append(val_.type_)
        #print('\tnext_kill_op = ', next_kill_op)
        print('total = ',total)
        #print('peak = ', peak)
    tokill_memo = 0
    for val in next_kill:
        tokill_memo += val.type_.memo_bytes
    #print("tokill_memo = ",tokill_memo)

    return peak /1024 /1024 

def overlap_input(op: Operation):
    """
    判断操作 op 的输出是否可以与其某个输入共享内存（即重叠）。
    如果可以，返回该输入的索引；否则返回 None。
    """
    # 不支持多输出操作
    if len(op.outputs_) > 1:
        return None

    out = op.outputs_[0]

    # 检查是否为逐元素操作或类型重解释操作
    if (op.op_.name_ not in ewOps) and (op.op_.name_ not in reinterpOps):
        return None

    # 遍历所有输入，寻找与输出大小相同且不是参数的输入
    for i, in_ in enumerate(op.inputs_):
        if in_.type_.rank == 0:
            continue
        if in_.type_.memo_bytes == out.type_.memo_bytes:
            return i

    return None

def compute_inc_dec(op: Operation, killed):
    """
    计算操作的内存增量变化
    :param op: 操作对象
    :param killed: 被释放的值列表
    :return: (内存增量, 内存减量) 元组
    """
    # 检查输出值是否可以覆盖输入值
    ovl_idx = overlap_input(op)
    if ovl_idx != None and op.inputs_[ovl_idx] not in killed:
        ovl_idx = None

    # 计算过渡到临时状态时的内存增量
    inc = 0
    if ovl_idx == None:
        inc = sum(val.type_.memo_bytes for val in op.outputs_)

    # 计算过渡到稳定状态时的内存减量
    ovl_val = op.inputs_[ovl_idx] if ovl_idx != None else None
    dec = 0
    for val in op.inputs_:
        #if val.kind == ValueKind.PARAM:  # 跳过参数
        #    continue
        if val not in killed:  # 值未被释放
            continue
        if val == ovl_val:  # 被覆盖的值不应计入
            continue
        dec += val.type_.memo_bytes

    return (inc, dec)

if __name__ == '__main__':
    i = 4
    with open(f'./ReBench/{i}/code.txt', 'r') as f:
        mod = relay.parse(f.read())

    static_mod = relay.transform.DynamicToStatic()(relay.transform.InferType()(mod))
    graph_mod = build_graph(static_mod)['main']
    graph_mod = preprocess_op_seq(graph_mod)
    print(graph_mod)
    visualize(graph_mod, 'without_tuple_mod', f'./ReBench/{i}/output')