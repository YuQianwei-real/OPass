from typing import List, Dict, Set, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from weakref import ref, WeakSet
import logging
import math

from ..graph.base import Graph, Operation, Value, Input, Output
from .hier import HierGraph, Sequence, Group, MemStateVec, HierVertex
from .mem import compute_inc_dec

# 假设的基础类型定义（需根据实际业务逻辑补充）

def AddUnique(lst: List[Any], item: Any) -> bool:
    """添加唯一元素"""
    if item not in lst:
        lst.append(item)
        return True
    return False

def Remove(lst: List[Any], item: Any) -> bool:
    """移除指定元素"""
    try:
        lst.remove(item)
        return True
    except ValueError:
        return False

def Contains(container, item) -> bool:
    """检查容器是否包含元素"""
    return item in container

class JoinVisitor:
    def __init__(self, hier: HierGraph):
        self.hier = hier
    
    def Join(self):
        """合并序列的主函数"""
        for vert in self.hier.inputs_:  # 需实现逆拓扑序遍历
            self.Visit(vert)
    
    def Visit(self, vert: HierVertex):
        """通用访问方法"""
        if isinstance(vert, Input):
            self.VisitInput(vert)
        elif isinstance(vert, Output):
            self.VisitOutput(vert)
        elif isinstance(vert, Sequence):
            self.VisitSequence(vert)
        else:
            raise ValueError(f"Unsupported vertex type: {type(vert)}")
    
    def VisitInput(self, input_vert: Input):
        """处理输入节点"""
        for succ in input_vert.succs:
            self.Visit(succ)
    
    def VisitOutput(self, output_vert: Output):
        """处理输出节点"""
        pass
    
    def VisitSequence(self, seq: Sequence):
        """处理序列节点"""
        # 初始化内存状态
        states = MemStateVec()
        inc, dec = self.computeIncDec(seq.oprs_[0])
        states.Append(inc, dec)
        
        # 迭代合并后续节点
        cur = seq
        while True:
            # 检查是否满足合并条件
            if len(cur.Succs) != 1 or not isinstance(cur.Succs[0], Sequence):
                break
            
            next_seq = cur.Succs[0]
            if len(next_seq.Preds) != 1:
                break
            
            # 尝试合并
            inc, dec = self.computeIncDec(next_seq.oprs_[0])
            s, t = states.Compute_state(inc, dec)
            
            if s > states.Latest or t > states.Latest:
                break  # 内存占用增加，停止合并
            
            # 执行合并
            states.Append(inc, dec)
            self.join(cur, next_seq)
            cur = next_seq
        
        # 继续处理后续节点
        for succ in seq.Succs:
            self.Visit(succ)
    
    @staticmethod
    def computeIncDec(op: Operation) -> Tuple[int, int]:
        """计算内存增量/减量"""
        killed = []
        for inp in op.inputs_:
            if all(use == op for use in inp.uses_):
                AddUnique(killed, inp)
        return compute_inc_dec(op, killed)  # 需要具体实现
    
    def join(self, prev: Sequence, next: Sequence):
        """合并两个序列"""
        # 修改序列数据
        for op in next.oprs_:
            prev.oprs_.append(op)
            self.hier.op_to_seq[op] = prev 
        
        prev.outputs_ = next.outputs_
        
        # 重新连接节点
        prev.Succs = next.Succs
        for succ in prev.Succs:
            for i, pred_ref in enumerate(succ.Preds):
                if pred_ref() == next:
                    succ.Preds[i] = prev
 

def JoinSequencePass(hier: HierGraph):
    """C++ Run方法的Python封装"""
    return  JoinVisitor(hier).Join()

class SeqPred:
    def __init__(self, func: Callable[[Sequence], bool]):
        self.func = func
    
    def __call__(self, seq: Sequence) -> bool:
        return self.func(seq)

class SequenceDetector:
    def __init__(
        self, 
        in_set: SeqPred,
        get_succs: Callable[[HierVertex], List[HierVertex]],
        set_: Set[Sequence],
        frontier: List[Sequence],
        sink: List[Sequence]
    ):
        self.in_set = in_set
        self.get_succs = get_succs
        self.set = set_
        self.frontier = frontier
        self.sink = sink
    
    def VisitSequence(self, seq: Sequence) -> bool:
        if not self.in_set(seq):
            return False
        
        self.set.add(seq)
        succs = self.get_succs(seq)
        
        is_frontier = False
        is_sink = True
        
        for succ in succs:
            not_in = not self.Visit(succ)
            is_frontier |= not_in
            is_sink &= not_in
        
        if is_frontier:
            AddUnique(self.frontier, seq)
        if is_sink:
            AddUnique(self.sink, seq)
        
        return True
    
    def Visit(self, vert: HierVertex) -> bool:
        if isinstance(vert, Sequence):
            return self.VisitSequence(vert)
        return False

def countConsumed(
    set_: Set[Sequence], in_front: List[Sequence]) -> List[Tuple[Value, int]]:
    """计算输入前沿消耗的值"""
    consumed = {}
    
    for seq in in_front:
        for inp in seq.inputs_:
            def_ = inp.def_.lock()
            if any(def_ in s.ops for s in set_):
                continue
            if inp in consumed:
                consumed[inp] += 1
            else:
                consumed[inp] = 1
    
    return list(consumed.items())

def countProduced(set_: Set[Sequence], out_front: List[Sequence]) -> List[Tuple[Value, int]]:
    """计算输出前沿产生的值"""
    produced = {}
    
    for seq in out_front:
        for out in seq.outputs:
            produced[out] = len(out.uses)
    
    # 移除组内消耗的值
    for seq in set_:
        for inp in seq.inputs:
            if inp in produced:
                produced[inp] -= 1
    
    # 过滤零计数
    return [(val, cnt) for val, cnt in produced.items() if cnt != 0]

def createGroup(
    set_: Set[Sequence],
    in_front: List[Sequence],
    out_front: List[Sequence],
    entrs: List[Sequence],
    exits: List[Sequence]
) -> Group:
    """创建组并调整连接"""
    group = Group()
    
    # 设置序列的组属性
    for seq in set_:
        seq.group = ref(group)
    
    # 初始化组属性
    group.seqs = list(set_)
    group.in_front = in_front
    group.out_front = out_front
    group.consumed = countConsumed(set_, in_front)
    group.produced = countProduced(set_, out_front)
    group.entrs = entrs
    group.exits = exits
    
    # 重新连接节点
    for front in in_front:
        front.preds = [p for p in front.preds if group.Contains(Sequence, p.lock())]
        
        for pred_ref in front.preds:
            pred = pred_ref()
            if not group.Contains(Sequence, pred):
                # 替换前驱
                HierVertex.ReplaceSuccOfPred(pred, front, group)
                AddUnique(group.preds, pred_ref)
    
    for front in out_front:
        front.succs = [s for s in front.succs if group.Contains(Sequence, s)]
        
        for succ in front.succs:
            if not group.Contains(Sequence, succ):
                # 替换后继
                HierVertex.ReplacePredOfSucc(succ, front, group)
                AddUnique(group.succs, succ)
    
    return group

class OutputSizeOptimizer:
    def __init__(self, all_seqs: Set[Sequence], root: Sequence):
        self.all_seqs = all_seqs
        self.root = root
        self.memo = {}
        self.best_set = []
        self.min_size = math.inf
    
    def Optimize(self) -> List[Sequence]:
        """执行动态规划优化"""
        # 构建前驱计数
        pred_count = {seq: len(seq.preds) for seq in self.all_seqs}
        pred_count[self.root] = 0
        
        # 开始搜索
        chosen = []
        succ_count = {}
        self.search(chosen, pred_count, succ_count)
        return self.best_set
    
    def search(
        self, 
        chosen: List[Sequence],
        pred_count: Dict[Sequence, int],
        succ_count: Dict[Sequence, int]
    ):
        """递归搜索最优解"""
        if tuple(chosen) in self.memo:
            return
        
        # 计算输出大小
        size = 0
        for seq in chosen:
            if succ_count.get(seq, 0) == 0:
                continue
            size += sum(out.type.Size() for out in seq.outputs)
        
        if size != 0:
            self.memo[tuple(chosen)] = size
            if (size < self.min_size or 
                (size == self.min_size and len(chosen) > len(self.best_set))):
                self.min_size = size
                self.best_set = chosen.copy()
        
        # 查找零前驱序列
        cand = [seq for seq, count in pred_count.items() if count == 0]
        
        # 尝试扩展解集
        for seq in cand:
            idx = len(chosen)
            chosen.append(seq)
            
            # 更新计数
            pred_count.pop(seq)
            for succ in seq.succs:
                if succ in pred_count:
                    pred_count[succ] -= 1
            
            succ_count[seq] = len(seq.succs)
            for pred in seq.Preds():
                if pred in succ_count:
                    succ_count[pred] -= 1
            
            # 递归搜索
            self.search(chosen, pred_count, succ_count)
            
            # 回溯
            chosen.pop()
            pred_count[seq] = 0
            for succ in seq.succs:
                if succ in pred_count:
                    pred_count[succ] += 1
            
            succ_count.pop(seq)
            for pred in seq.Preds():
                if pred in succ_count:
                    succ_count[pred] += 1

def MakeGroupPass(hier: HierGraph):
    """MakeGroupPass的主函数"""
    # 构建支配树
    if not hier.inputs_:
        logging.error("Input list of the hierarchical graph is empty.")
        return
    
    # 构建支配树（需具体实现）
    dom_nodes = DomBuilder(HierVertex).Build(hier.inputs_[0])  # 需要DomBuilder实现
    for node in dom_nodes:
        if vertex := node.vertex.lock():
            vertex.dom = node
    
    # 构建后支配树
    if not hier.outputs:
        logging.error("Output list of the hierarchical graph is empty.")
        return
    
    post_dom_nodes = DomBuilder(HierVertex).Build(hier.outputs[0])  # 同上
    for node in post_dom_nodes:
        if vertex := node.vertex.lock():
            vertex.postDom = node
    
    # 查找所有cell输出
    cell_outs = []
    for vert in RpoHierRange(hier):  # 需要逆拓扑序遍历
        vert.BackupEdges()
        if isinstance(vert, Sequence) and isCellOut(vert):
            cell_outs.append(vert)
    
    # 从cells构建组
    for out in cell_outs:
        if out.group and out.group.lock():
            continue
        makeGroupFromCell(out)

def isCellOut(seq: Sequence) -> bool:
    """判断是否为cell输出(示例实现)"""
    return seq.oprs_ and seq.oprs_[0].op_.name_ == "concat"

def makeGroupFromCell(cell_out: Sequence):
    """基于cell创建组"""
    # 检测输入前沿
    seqs = set()
    cell_in_front = []
    cell_entrances = []
    
    detector = SequenceDetector(
        lambda seq: cell_out.PostDominates(seq),
        lambda v: v.Preds(),
        seqs,
        cell_in_front,
        cell_entrances
    )
    detector.Visit(cell_out)
    
    # 检测输出前沿
    intruded = set()
    intr_out_front = []
    intr_exits = []
    
    detector = SequenceDetector(
        lambda seq: cell_out.Dominates(seq),
        lambda v: v.Succs(),
        intruded,
        intr_out_front,
        intr_exits
    )
    detector.Visit(cell_out)
    
    # 直接创建组（条件判断）
    if not MakeGroupPass.makeCell or cell_out in intr_out_front:
        createGroup(seqs, cell_in_front, [cell_out], cell_entrances, [cell_out])
        return
    
    # 动态规划优化
    optimizer = OutputSizeOptimizer(intruded, cell_out)
    min_size_set = optimizer.Optimize()
    
    if len(min_size_set) <= 2:
        createGroup(seqs, cell_in_front, [cell_out], cell_entrances, [cell_out])
        return
    
    # 调整组范围
    intruded.clear()
    intr_out_front.clear()
    intr_exits.clear()
    
    detector = SequenceDetector(
        lambda seq: seq in min_size_set,
        lambda v: v.Succs(),
        intruded,
        intr_out_front,
        intr_exits
    )
    detector.Visit(cell_out)
    intruded.discard(cell_out)
    
    # 查找输入前沿
    intr_in_front = []
    intr_entrances = []
    
    for succ in cell_out.succs:
        if isinstance(succ, Sequence):
            intr_in_front.append(succ)
            if not any(
                Contains(intruded, Cast(Sequence, pred.lock())) 
                for pred in succ.preds
            ):
                intr_entrances.append(succ)
    
    # 创建组
    createGroup(seqs, cell_in_front, [cell_out], cell_entrances, [cell_out])
    createGroup(intruded, intr_in_front, intr_out_front, intr_entrances, intr_exits)