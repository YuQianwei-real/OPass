from __future__ import annotations
import logging
import random
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict

from ..graph.base import Vertex, Graph, Operation, Value, VertexKind
from .hier import HierVertex, HierGraph, MemStateVec, Group, Sequence, HierKind, HierInput, HierOutput
from .join import JoinSequencePass, MakeGroupPass
from .life import compute_op_lifetimes_networkx
from .mem import estimate_peak_hmcos0, compute_inc_dec
from .util import AddUnique, Remove

# 假设基础数据结构已定义（如 OpRef, Graph, MemStateVec 等）
# 此处仅展示核心逻辑框架

# 日志初始化
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

max_budget = 2**63 -1

def extractZeroIn(predCnt: Dict[Vertex, int], zeroPred: List[Vertex]):
    """提取入度为0的节点"""
    for vert, cnt in predCnt.items():
        if cnt == 0:
            zeroPred.append(vert)
    for op in zeroPred:
        predCnt.pop(op, None)
    return predCnt, zeroPred

def sampleVertex(predCnt: Dict[Vertex, int], zeroPred: List[Vertex], rng: random.Random):
    """随机选择一个零入度节点并更新依赖关系"""
    vert = rng.choice(zeroPred)
    zeroPred.remove(vert)
    for succ in vert.succs:
        if isinstance(succ, Vertex):
            predCnt[succ] -= 1
    extractZeroIn(predCnt, zeroPred)
    return vert

def RandomSample(graph: 'Graph', rng: random.Random) -> List[Operation]:
    """随机生成调度顺序"""
    predCnt = {op: len(op.preds) for op in graph.oprs_}
    for input in graph.inputs_:
        for succ in input.succs:
            predCnt[succ] -= 1
    zeroPred = []
    extractZeroIn(predCnt, zeroPred)
    sched = []
    while zeroPred:
        vert= sampleVertex(predCnt, zeroPred, rng)
        sched.append(vert)
    return sched

class SchedResult:
    """
    Represents the result of a scheduling operation.
    """
    def __init__(self, seq: List[Operation] = None, states: MemStateVec = None):
        """
        Initializes a SchedResult object.

        Args:
            seq (list, optional): Scheduled sequence of ops. Defaults to None (creates empty list).
            states (MemStateVec, optional): Memory states of the scheduled sequence. Defaults to None (creates invalid result).
        """
        if seq is None and states is None:
            # Default constructor: invalid result
            self.valid = False
            self.seq = []
            self.states = MemStateVec() # Or a default invalid MemStateVec instance
        else:
            # Constructor with sequence and states: valid result
            self.valid = True
            self.seq = seq if seq is not None else []
            self.states = states if states is not None else MemStateVec()# Assumes states is a MemStateVec object

    def Extend(self, other: SchedResult):
        """
        Extends this result with another SchedResult.

        Args:
            other (SchedResult): The result to extend with.
        """
        self.seq.extend(other.seq)
        
        self.states.Extend(other.states)
        
    def Print(self):
        """
        Prints the scheduled sequence and states.
        """
        # Assumes ZipRange behaves like zip [[10]]
        if self.seq and self.states:
            try:
                for op, state in zip(self.seq, self.states):
                    # Assumes op has a 'type' attribute and state is indexable
                    print(f"{op.op_.name_:<18} {state[0]:>8}^ {state[1]:>8}_")
            except (AttributeError, IndexError, TypeError) as e:
                 print(f"Error printing result: {e}. Check op.type or state format.")
        else:
            print("Invalid or empty schedule result.")


class PartialSchedResult(SchedResult):
    """
    Represents a partial scheduling result, extending SchedResult.
    """

    def __init__(self, seq: List[Operation] = None, states:MemStateVec = None, 
                 pred_cnt: Dict[HierVertex, int]=None, use_cnt: Dict[Value, int] =None):
        """
        Initializes a PartialSchedResult object.
        Args:
            seq (list, optional): Scheduled sequence of ops. Defaults to None.
            states (MemStateVec, optional): Memory states. Defaults to None.
            pred_cnt (dict, optional): Predecessor count map (HierVertRef -> int). Defaults to None (creates empty dict).
            use_cnt (dict, optional): Use count map (ValueRef -> int). Defaults to None (creates empty dict).
        """
        # Initialize base class (SchedResult)
        super().__init__(seq, states)

        # Initialize additional members for partial results
        # Assumes std::unordered_map translates to Python dict [[9]]
        self.pred_cnt = pred_cnt if pred_cnt is not None else {}
        self.use_cnt = use_cnt if use_cnt is not None else {}

    def update(self, other: PartialSchedResult):
        """
        Updates this result with another if the other has a lower peak memory state.

        Args:
            other (PartialSchedResult): The other result to potentially update from.
        """
        # Assumes MemStateVec has a 'peak' method [[9]]
        if other.states.Peak() < self.states.Peak():

            # Swap sequences (lists)
            self.seq, other.seq = other.seq, self.seq

            # Swap states (MemStateVec objects)
            self.states.Swap(other.states)

            # Swap predecessor counts (dicts)
            self.pred_cnt, other.pred_cnt = other.pred_cnt, self.pred_cnt

            # Swap use counts (dicts)
            self.use_cnt, other.use_cnt = other.use_cnt, self.use_cnt


class GroupContext:
    """组调度上下文"""
    def __init__(self, group: Group, id:int ,use_cnt: Dict[Any, int]):
        self.id = id
        self.group = group
        self.kill = [1 if pair[1] == use_cnt[pair[0]] else 0 for pair in group.consumed]
    
    def __hash__(self):
        return self.id

    def __eq__(self, other: 'GroupContext') -> bool:
        return self.group == other.group and self.kill == other.kill

def scheduleSequence(seq: Sequence, use_cnt: Dict[Value, int], budget: int) -> SchedResult:
    """调度单个序列，计算内存状态"""
    states = MemStateVec()
    for op in seq.oprs_:
        killed = []
        for val in op.inputs_:
            if val.type_.memo_bytes < 5:
                continue
            if val not in use_cnt:
                killed.append(val)
            else:
                use_cnt[val] -= 1
                if use_cnt[val] == 0:
                    killed.append(val)

        inc, dec = compute_inc_dec(op, killed)
        s, t = states.Compute_state(inc, dec)
        if s > budget:
            return SchedResult()
        states.Append(inc, dec)
        for val in killed:
            use_cnt.pop(val, None)
        for val in op.outputs_:
            use_cnt[val] = len(val.uses_)
    return SchedResult(seq = seq.oprs_, states = states)

def scheduleGroupRpo(group: Group, use_cnt: Dict[Any, int], budget: int) -> SchedResult:
    """按反向后序遍历调度组"""
    opSeq = []
    states = MemStateVec()
    for vert in reversed(group.exits):
        assert isinstance(vert, Sequence), "Object is not of type `Sequence`"
        result = scheduleSequence(vert, use_cnt, budget - states.Latest())
        if not result.valid:
            return result
        opSeq.extend(result.seq)
        states.Extend(result.states)
    return SchedResult(opSeq, states)

#Use DP algorithm to schedule the group
def scheduleGroupDp(group: Group, use_cnt: Dict[Value, int], budget)-> SchedResult:
    #Initialize predecessor count of sequences inside group
    pred_cnt: Dict[Sequence, int] = {}
    for seq in group.seqs:
        pred_cnt[seq] = len(seq.preds)
    #Initialize memoization map
    zero_in: List[Sequence] = []
    pred_cnt, zero_in = extractZeroIn(pred_cnt, zero_in)
    memo: Dict[List[HierVertex], PartialSchedResult] = {}
    memo[tuple(zero_in)] = PartialSchedResult(seq = [], states = MemStateVec(), pred_cnt = pred_cnt, use_cnt = use_cnt)
    #Iterate |V| steps
    for i in range(len(group.seqs)):
        new_memo = {}
        for zero_in, result in memo.items():
            zero_in = list(zero_in)
            for vert in zero_in:
                use_cnt = result.use_cnt.copy()
                remaining_budget = budget - result.states.Latest()
                vert_result = scheduleSequence(vert, use_cnt, remaining_budget)
                new_memo = update_result(vert, zero_in, result, vert_result, use_cnt, new_memo)
        if not new_memo:
            return SchedResult()
        memo = new_memo

    return memo.get(tuple(), SchedResult())

def update_result(vert: HierVertex, zero_in: List[HierVertex], result: PartialSchedResult, 
                  vert_result: SchedResult, use_cnt: Dict[Value, int], new_memo):
    if not vert_result.valid:
        return None

    new_seq = result.seq + vert_result.seq #Extend op sequence
    
    result.states.Extend(vert_result.states) # Extend memory states
    # Update zero-indegree set
    pred_cnt = result.pred_cnt.copy()
    for suc in vert.succs:
        pred_cnt[suc] -= 1
    new_zeroIn = zero_in.copy()
    new_zeroIn.remove(vert)
    pred_cnt, new_zeroIn = extractZeroIn(pred_cnt, new_zeroIn)
    #Memoize this partial result
    new_result = PartialSchedResult(new_seq, result.states, pred_cnt, use_cnt)
    if tuple(new_zeroIn) in new_memo:
        new_memo[tuple(new_zeroIn)].update(new_result)
    else:
        new_memo[tuple(new_zeroIn)] = new_result
    #new_memo[tuple(new_zeroIn)] = new_result

    return new_memo

def updateGroupUseCount(group: Group, use_cnt: Dict['Value', int]):
    """更新组的使用计数"""
    killed = []
    for val, num in enumerate(group.consumed):
        if val not in use_cnt:
            killed.append(val)
            continue
        use_cnt[val] -= num
        if use_cnt[val] == 0:
            killed.append(val)
    for val in killed:
        use_cnt.pop(val, None)
    for pair in group.produced:
        use_cnt[pair[0]] = pair[1]
    return use_cnt


class HierScheduler:
    """分层调度器"""
    def __init__(self, hier: 'HierGraph', budget: int, groupMemo: Dict[GroupContext, SchedResult]):
        self.hier = hier
        self.budget = budget
        self.groupMemo = groupMemo
        self.group_id = 0

    def Schedule(self) -> List['Operation']:
        predCnt:Dict[HierVertex, int] = {}
        for vert in self.hier.DfsHierRange():
            if isinstance(vert, HierInput) or isinstance(vert, HierOutput):
                continue
            if isinstance(vert, HierVertex):
                predCnt[vert] = len(vert.preds)

        use_cnt: Dict[Value, int] = {}
        for input in self.hier.hierIns_:
            for succ in input.succs:
                predCnt[succ] -= 1
            val = input.in_.value_
            use_cnt[val] = len(val.uses_)

        zeroIn: List[HierVertex] = []
        extractZeroIn(predCnt, zeroIn)
        #predCnt, zeroIn = extractZeroIn(predCnt, zeroIn)
        initSize = sum(input.in_.value_.type_.memo_bytes for input in self.hier.hierIns_)
        memo: Dict[Set[HierVertex], PartialSchedResult] = {}
        memo[tuple(zeroIn)] = PartialSchedResult(seq=[], states=MemStateVec(initSize), pred_cnt=predCnt, use_cnt=use_cnt)
        print(predCnt)
        for i in range(len(predCnt)):
        #Iterate each partial result and build partial schedule with one more vertex
            newMemo = {}
            for zeroInKey, result in memo.items():
                for vert in list(zeroInKey):
                    useCntCopy = result.use_cnt.copy()
                    vertResult = self.scheduleVertex(vert, useCntCopy, result.states)
                    if vertResult == None:
                        continue
                    newMemo = update_result(vert, list(zeroInKey), result, vertResult, useCntCopy, newMemo)
            assert len(newMemo) > 0
            memo, newMemo = newMemo, memo
            #print(memo)
        return memo[tuple()].seq

    def scheduleVertex(self, vert: HierVertex, use_cnt: Dict[Any, int], prevStates: 'MemStateVec') -> SchedResult:
        localBudget = self.budget - prevStates.Latest()
        if isinstance(vert, Sequence):
            return scheduleSequence(vert, use_cnt, localBudget)
        elif isinstance(vert, Group):
            #Check if there is memoized result
            group = vert
            ctx = GroupContext(group, self.group_id, use_cnt)
            self.group_id += 1
            if ctx in self.groupMemo:
                result = self.groupMemo[ctx]
                if result.states.Peak() > localBudget:
                    return SchedResult()
                use_cnt = updateGroupUseCount(group, use_cnt)
                return result
            # Try schedule using reverse post-order
            rpoUseCnt = use_cnt.copy()
            rpoBudget = min(localBudget, prevStates.Peak() - prevStates.Latest())
            rpoResult = scheduleGroupRpo(group, rpoUseCnt, rpoBudget)
            if rpoResult.valid:
                use_cnt, rpoUseCnt = rpoUseCnt, use_cnt
                return rpoResult
            #Schedule group using DP and memoize the result
            dpResult = scheduleGroupDp(group, use_cnt, localBudget)
            if not dpResult.valid:
                return SchedResult()
            updateGroupUseCount(group, use_cnt)
            self.groupMemo[ctx] = dpResult
            return dpResult
        else:
            return None
            raise ValueError("Invalid vertex kind")



def find_edges_to_restore(frontier: List['Sequence'], neighbors: List[Vertex],
    get_neighbor_prev: List['Sequence'], get_neighbor_frontier: List['Sequence']
    )-> Dict['Sequence', List[Vertex]]:
    
    restore_map = {seq: [] for seq in frontier}
    
    for vert in neighbors:
        if isinstance(vert, Group):
            # Handle group type
            for front_seq in get_neighbor_frontier(vert):
                for out_seq in get_neighbor_prev(front_seq):
                    if isinstance(out_seq, Sequence) and out_seq in restore_map:
                        restore_map[out_seq].append(vert)
        else:
            for out_seq in get_neighbor_prev(vert):
                if isinstance(out_seq, Sequence) and out_seq in restore_map:
                    restore_map[out_seq].append(vert)
    return restore_map

def ungroup(group: Group):
    # Reconnect predecessors with input frontiers
    in_restore = find_edges_to_restore(
        group.inFront,
        group.preds,  # Assuming pred contains predecessors
        lambda vert: vert.prevSuccs,  # Assuming method exists
        lambda grp: grp.outFront
    )
    
    for front, restores in in_restore.items():
        for neighbor in restores:
            AddUnique(front.preds, neighbor)
            Remove(neighbor.succs, group)
            AddUnique(neighbor.succs, front)
    
    # Reconnect successors with output frontiers
    out_restore = find_edges_to_restore(
        group.outFront,
        group.succs,  # Assuming succ contains successors
        lambda vert: vert.prevPreds(),  
        lambda grp: grp.inFront
    )
    
    for front, restores in out_restore.items():
        for neighbor in restores:
            AddUnique(front.succs, neighbor)
            Remove(neighbor.preds, group)
            AddUnique(neighbor.preds, front)
    
    # Clean up group references
    for seq in group.seqs:
        seq.group_ = None

def tryUngroupSucc(seq: Sequence):
    changed = False
    while (True):
        iterChanged = False
        for suc in seq.succs:
            if isinstance(suc, Group):
                ungroup(suc)
                iterChanged = True
                changed = True
                break
        if not iterChanged:
            break
    return changed


def HierarchicalSchedule(graph: Graph) -> List[Operation]:
    """分层调度主函数"""

    hier = HierGraph(graph)
    JoinSequencePass(hier)
    MakeGroupPass(hier)

    groupMemo: Dict[GroupContext, SchedResult] = {} 
    lastSched: List[Operation] = [] 
    lastPeak = float('inf') #max_budget
    while True:
        sched = HierScheduler(hier, lastPeak, groupMemo).Schedule()
        peak, peakValues = estimate_peak_hmcos0(sched, graph)
        if not peakValues:
            break

        if peak < lastPeak:
            lastPeak = peak
            lastSched = sched
        #Locate sequences related to this peak
        relSeqs: Set[Sequence] = set()
        for val in peakValues:
            assert isinstance(val, Value), "val is not Value class"
            relSeqs.add(hier.op_to_seq[val.def_])
        #ungroup
        changed = False
        for seq in relSeqs:
            assert isinstance(seq, Sequence)
            group = seq.group_
            if group:
                ungroup(group)
                changed = True
            changed |= tryUngroupSucc(seq)
        if not changed:
            break
    return lastSched

def sampleGroupPeak(group: Group, use_cnt: Dict[Value, int], rng):
    pred_cnt: Dict[Sequence, int] = {}
    for seq in group.seqs:
        pred_cnt[seq] = len(seq.preds)
    
    zero_in: List[Sequence] = []
    extractZeroIn(pred_cnt, zero_in)

    sched = []
    states = MemStateVec()
    while len(pred_cnt) != 0:
        seq = sampleVertex(pred_cnt, zero_in, rng)
        result = scheduleSequence(seq, use_cnt, max_budget)
        sched.extend(result.seq)
        states.Extend(result.states)

    return states.Peak()


def SerenitySchedule(graph: 'Graph', joinOps: bool = True, trySimple: bool = True, nSamples: int = 10) -> List['Operation']:
    """主调度函数，结合多种策略"""
    hier = HierGraph(graph)
    if joinOps:
        JoinSequencePass(hier)
    MakeGroupPass(hier)

    topVerts = [v for v in hier.RpoHierRange()]
    sched = []
    states = MemStateVec()
    use_cnt = {}
    for i, vert in enumerate(topVerts):
        #logger.info(f"Scheduling vertex {i+1}/{len(topVerts)}")
        if isinstance(vert, HierInput):
            input = vert.in_
            use_cnt[input.value_] = len(input.value_.uses_)
            states = MemStateVec(input.value_.type_.memo_bytes)
        elif isinstance(vert, Sequence):
            for inp in vert.inputs_:
                use_cnt[inp] = len(inp.uses_) 
            result = scheduleSequence(vert, use_cnt, max_budget)
            sched.extend(result.seq)
            states.Extend(result.states)
        elif isinstance(vert, Group):

            group = vert
            if trySimple:
                rpoUseCnt = use_cnt.copy()
                rpoResult = scheduleGroupRpo(group, rpoUseCnt, states.Peak() - states.Latest())
                if rpoResult.valid:
                    use_cnt = rpoUseCnt
                    sched.extend(rpoResult.seq)
                    states.Extend(rpoResult.states)
                    continue
            budget = max_budget
            rng = random.Random()
            #logger.info("Sampling schedules.")
            for _ in range(nSamples):
                budget = min(budget, sampleGroupPeak(group, use_cnt, rng))
            #logger.info(f"Scheduling group with budget {budget / 1024} KB.")
            dpResult = scheduleGroupDp(group, use_cnt, budget)
            sched.extend(dpResult.seq)
            states.Extend(dpResult.states)
        elif isinstance(vert, HierOutput):
            continue
    return sched