from typing import List, Dict, Set, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from weakref import ref, WeakSet
import logging
import math, bisect

from ..graph.base import Graph, Operation, Value, Input, Output, Vertex, Constant, Global
from .hier import HierGraph, Sequence, Group, MemStateVec, HierVertex, HierInput, HierOutput
from .mem import compute_inc_dec
from .dom import DomBuilder 
from .util import replace_pred_of_all_succs, AddUnique, replace_pred_of_succ, replace_succ_of_pred


# 假设的基础类型定义（需根据实际业务逻辑补充）
makeCell = True

class JoinVisitor:
    def __init__(self, hier: HierGraph):
        self.hier = hier
    
    def Join(self):
        """合并序列的主函数"""
        for vert in self.hier.hierIns_:  # 需实现逆拓扑序遍历
            self.Visit(vert)
    
    def Visit(self, vert: HierVertex):
        """通用访问方法"""
        if isinstance(vert, HierInput):
            self.VisitInput(vert)
        elif isinstance(vert, HierOutput):
            self.VisitOutput(vert)
        elif isinstance(vert, Sequence):
            self.VisitSequence(vert)
        elif isinstance(vert, Group):
            pass
        else:
            raise ValueError(f"Unsupported vertex type: {type(vert)}")
    
    def VisitInput(self, input_vert: HierInput):
        """处理输入节点"""
        for succ in input_vert.succs:
            self.Visit(succ)
    
    def VisitOutput(self, output_vert: HierOutput):
        """处理输出节点"""
        return 

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
            if len(cur.succs) != 1 or not isinstance(cur.succs[0], Sequence):
                break
            
            next_seq = cur.succs[0]
            if len(next_seq.preds) != 1:
                break
            
            # 尝试合并
            inc, dec = self.computeIncDec(next_seq.oprs_[0])
            s, t = states.Compute_state(inc, dec)
            
            if s > states.Latest() or t > states.Latest():
                break  # 内存占用增加，停止合并
            
            # 执行合并
            states.Append(inc, dec)
            self.join(cur, next_seq)
            #cur = next_seq
            
        
        # 继续处理后续节点
        for succ in seq.succs:
            self.Visit(succ)
    
    @staticmethod
    def computeIncDec(op: Operation) -> Tuple[int, int]:
        """计算内存增量/减量"""
        killed: List[Value] = []
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
        prev.succs = next.succs
        replace_pred_of_all_succs(next, prev)
        """
        for succ in prev.succs:
            for i, pred_ref in enumerate(succ.preds):
                if pred_ref == next:
                    succ.preds[i] = prev
        """            
        self.hier.sequences_.remove(next)
 

def JoinSequencePass(hier: HierGraph):
    """C++ Run方法的Python封装"""
    return  JoinVisitor(hier).Join()


class SequenceDetector:
    def __init__(
        self, 
        in_set: Callable[[HierVertex], List[HierVertex]],
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

def countConsumed(set_: Set[Sequence], in_front: List[Sequence]) -> List[Tuple[Value, int]]:
    """计算输入前沿消耗的值"""
    consumed = {}
    
    for seq in in_front:
        for inp in seq.inputs_:
            def_ = inp.def_
            if any(def_ in s.oprs_ for s in set_):
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
        for out in seq.outputs_:
            produced[out] = len(out.uses_)
    
    # 移除组内消耗的值
    for seq in set_:
        for inp in seq.inputs_:
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
        seq.group_ = group
    
    # 初始化组属性
    group.seqs = list(set_)
    group.inFront = in_front
    group.outFront = out_front
    group.consumed = countConsumed(set_, in_front)
    group.produced = countProduced(set_, out_front)
    group.entrs = entrs
    group.exits = exits
    
    # 重新连接节点
    for front in in_front:
        new_preds: List[Sequence] = []
        
        for pred in front.preds:
            if group.Contains(pred):
                new_preds.append(pred) #keep this predecessor as it is in the group
            else:
                # 替换前驱
                replace_succ_of_pred(pred, front, group)
                AddUnique(group.preds, pred)

        front.preds = new_preds
           
    
    for front in out_front:
        new_succs: List[Sequence] = []
        
        for succ in front.succs:
            if group.Contains(succ):
                new_succs.append(succ)
            else:
                # 替换后继
                replace_pred_of_succ(succ, front, group)
                AddUnique(group.succs, succ)
        
        front.succs = new_succs
    
    return group

#Use DP to find a subset of intruded sequences which minimize size of its outputs.
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
        chosen: List[Sequence]= []
        succ_count: Dict[Sequence, int] = {}
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
            size += sum(out.type_.memo_bytes for out in seq.outputs_)
        
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
            idx = bisect.bisect_left(chosen, seq)
            chosen.insert(idx, seq)
            #chosen.append(seq)
            # 更新计数
            pred_count.pop(seq)
            for succ in seq.succs:
                if succ in pred_count:
                    pred_count[succ] -= 1
            
            succ_count[seq] = len(seq.succs)
            for pred in seq.preds:
                if pred in succ_count:
                    succ_count[pred] -= 1
            
            # 递归搜索
            self.search(chosen, pred_count, succ_count)
            
            # 回溯
            chosen.pop(idx)
            #chosen.remove(seq)
            pred_count[seq] = 0
            for succ in seq.succs:
                if succ in pred_count:
                    pred_count[succ] += 1
            
            succ_count.pop(seq)
            for pred in seq.preds:
                if pred in succ_count:
                    succ_count[pred] += 1
    

def MakeGroupPass(hier: HierGraph):
    """MakeGroupPass的主函数"""
    # 构建支配树
    if len(hier.hierIns_) == 0:
        logging.error("Input list of the hierarchical graph is empty.")
        return
    
    # 构建支配树（需具体实现）
    dom_nodes = DomBuilder(hier).Build(hier.root)  # 需要DomBuilder实现
    for node in dom_nodes:
        node.hierVertex_.dom = node
    
    # 构建后支配树
    if not hier.hierOuts_:
        logging.error("Output list of the hierarchical graph is empty.")
        return
    elif len(hier.hierOuts_) > 1:
        logging.warning("Post-dominator tree will only be built for the first output vertex.")

    post_dom_nodes = DomBuilder(hier).Build(hier.hierOuts_[0])  # 同上
    for node in post_dom_nodes:
        node.hierVertex_.postDom = node
    
    # 查找所有cell输出.以逆后序方式查找所有单元格输出，同时备份前驱和后继
    cell_outs = []
    for vert in hier.RpoHierRange():  # 需要逆拓扑序遍历

        if isinstance(vert, Sequence) and isCellOut(vert):
            cell_outs.append(vert)
    
    # 从cells构建组
    for out in cell_outs:
        if isinstance(out, Sequence) and out.group_ == None:
            makeGroupFromCell(out)

def isCellOut(seq: Sequence) -> bool:
    """判断是否为cell输出(示例实现)"""
    return seq.oprs_ and seq.oprs_[0].op_.name_ == "concatenate" #concatenate

def makeGroupFromCell(cell_out: Sequence):
    """基于cell创建组"""
    # 检测输入前沿
    seqs = set()
    cell_in_front: List[Sequence] = []
    cell_entrances: List[Sequence] = []
    
    detector = SequenceDetector(
        lambda seq: cell_out.PostDominates(seq),
        lambda v: v.preds,
        seqs,
        cell_in_front,
        cell_entrances
    )
    detector.Visit(cell_out)
    seqs = detector.set
    cell_in_front = detector.frontier
    cell_entrances = detector.sink
    
    # 检测输出前沿
    intruded: Set[Sequence] = set()
    intr_out_front: List[Sequence] = []
    intr_exits: List[Sequence] = []
    
    detector = SequenceDetector(
        lambda seq: cell_out.Dominates(seq),
        lambda v: v.succs,
        intruded,
        intr_out_front,
        intr_exits
    )
    detector.Visit(cell_out)
    introded = detector.set
    intr_out_front = detector.frontier
    intr_exits = detector.sink
    
    # 直接创建组（条件判断）
    if not makeCell or cell_out in intr_out_front:
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
        lambda v: v.succs,
        intruded,
        intr_out_front,
        intr_exits
    )
    detector.Visit(cell_out)
    intruded = detector.set
    intr_out_front = detector.frontier
    intr_exits = detector.sink

    intruded.discard(cell_out)
    
    # 查找输入前沿
    intr_in_front: List[Sequence] = []
    intr_entrances: List[Sequence] = []
    
    for succ in cell_out.succs:
        if isinstance(succ, Sequence):
            intr_in_front.append(succ)
            if not any(isinstance(pred, Sequence)
                and pred in intruded for pred in succ.preds
            ):
                intr_entrances.append(succ)
    
    # 创建组
    createGroup(seqs, cell_in_front, [cell_out], cell_entrances, [cell_out])
    createGroup(intruded, intr_in_front, intr_out_front, intr_entrances, intr_exits)