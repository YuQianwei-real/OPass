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

class Lifetime:
    # 静态常量：输入时间和未知时间
    TIME_INPUT = -1
    TIME_UNKNOWN = float('inf')  # 模拟 最大值

    def __init__(self, value, gen: float, kill: float):
        self.value = value  # Value 对象(Operation or Value)
        self.gen = gen      # 生命周期起始时间
        self.kill = kill    # 生命周期结束时间

    def Length(self):
        """计算生命周期长度"""
        return self.kill - self.gen  

    def Print(self):
        """打印生命周期信息"""
        print(f"{self.gen}:{self.kill} {self.value.name}")  # [[1]]


def compute_lifetime_hmcos(op_seq: List[Operation], graph :Graph):
    # Op sequence must be a full permutation of ops in graph
    assert len(op_seq) == len(graph.oprs_)

    # Initialize lifetime and use count of inputs
    val_life = {}
    use_cnt = {}
    for in_ in graph.inputs_:
        val = in_.value_
        val_life[val] = Lifetime(val, Lifetime.TIME_INPUT, Lifetime.TIME_UNKNOWN)
        use_cnt[val] = len(val.uses_)

    # Compute lifetime
    for i, op in enumerate(op_seq):
        # Initialize lifetime of its outputs
        for out in op.outputs_:
            val_life[out] = Lifetime(out, i, Lifetime.TIME_UNKNOWN)
            use_cnt[out] = len(out.uses_)

        # Compute lifetime ending of its inputs
        ovl_idx = overlap_input(op)
        for j, in_ in enumerate(op.inputs):
            if in_.kind == ValueKind.PARAM:
                continue
            if in_ not in use_cnt:
                raise ValueError(f"Value {in_.name} used without definition before.")
            cnt = use_cnt[in_]
            cnt -= 1
            use_cnt[in_] = cnt
            # If output can overlap this input, its life ends before this op.
            # Otherwise, it must keep alive until computation of this op is finished.
            if cnt == 0:
                val_life[in_].kill = i if ovl_idx == j else i + 1
                del use_cnt[in_]

    # Finalize lifetime of outputs
    end_time = len(op_seq)
    for out in graph.outputs:
        val_life[out.value].kill = end_time

    # Sort lifetime
    blocks = sorted([lt for lt in val_life.values()], key=cmp_by_gen_kill)

    return LifetimeStat((Lifetime.TIME_INPUT, end_time), blocks)

def compute_op_lifetimes_networkx(G: Graph, schedule: List[Operation]) -> Dict[str, Tuple[int, int]]:
    """
    计算每个操作符的生命周期(alloc_time, free_time)
    
    :param G: gencog有向图
    :param schedule: operation节点操作符调度顺序列表(如['op1', 'op2', ...])
    :return: 字典，键为操作符名称，值为(alloc_time, free_time)
    """
    assert len(schedule) == len(Graph.oprs_)

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

if __name__ == '__main__':
    i = 4
    with open(f'./ReBench/{i}/code.txt', 'r') as f:
        mod = relay.parse(f.read())

    static_mod = relay.transform.DynamicToStatic()(relay.transform.InferType()(mod))
    graph_mod = build_graph(static_mod)['main']
    graph_mod = preprocess_op_seq(graph_mod)
    print(graph_mod)
    visualize(graph_mod, 'without_tuple_mod', f'./ReBench/{i}/output')