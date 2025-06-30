import logging
import random
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict

from ..graph.base import Vertex, Graph, Operation, Value, VertexKind
from .hier import HierVertex, HierGraph, MemStateVec
from .join import AddUnique, Remove, JoinSequencePass, MakeGroupPass


# 假设基础数据结构已定义（如 OpRef, Graph, MemStateVec 等）
# 此处仅展示核心逻辑框架

# 日志初始化
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def PlotSchedule(sched: List['Operation'], graph: 'Graph', dir: str, name: str, format: str):
    """绘制调度顺序的图结构，使用 DOT 格式"""
    assert len(sched) == len(graph.oprs_)
    # 使用 graphviz 等库实现图渲染
    # 示例：创建节点和边并保存为文件
    pass

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
    predCnt, zeroPred = extractZeroIn(predCnt, zeroPred)
    return vert, zeroPred

def RandomSample(graph: 'Graph', rng: random.Random) -> List[Operation]:
    """随机生成调度顺序"""
    predCnt = {op: len(op.preds) for op in graph.oprs_}
    for input in graph.inputs_:
        for succ in input.succs:
            predCnt[succ] -= 1
    zeroPred = []
    predCnt, zeroPred = extractZeroIn(predCnt, zeroPred)
    sched = []
    while zeroPred:
        vert, zeroPred = sampleVertex(predCnt, zeroPred, rng)
        sched.append(vert)
    return sched

def ReversePostOrder(graph: Graph) -> List[Operation]:
    """按反向后序遍历生成调度顺序"""
    return [v for v in graph.rpo_vert_range() if isinstance(v, Operation)]

@dataclass
class SchedResult:
    """调度结果基类"""
    valid: bool = False
    seq: List[Operation] = []
    states: MemStateVec = None

    def Extend(self, other: SchedResult):
        self.seq.extend(other.seq)
        self.states.extend(other.states)

    def Print(self):
        for op, state in zip(self.seq, self.states):
            logger.info(f"{op.type:<18} {state.first:>8}^ {state.second:>8}_")

class PartialSchedResult(SchedResult):
    """部分调度结果，包含中间状态"""
    def __init__(self):
        super().__init__()
        self.pred_cnt = {}
        self.use_cnt = {}

class GroupContext:
    """组调度上下文"""
    def __init__(self, group, use_cnt: Dict[Any, int]):
        self.group = group
        self.kill = [use_cnt[val] == len(val.uses_) for val in group.consumed]

    def __eq__(self, other: 'GroupContext') -> bool:
        return self.group == other.group and self.kill == other.kill

def scheduleSequence(seq, use_cnt: Dict[Any, int], budget: int) -> SchedResult:
    """调度单个序列，计算内存状态"""
    states = MemStateVec()
    for op in seq.ops:
        killed = []
        for val in op.inputs:
            if val.kind != ValueKind.PARAM:
                use_cnt[val] -= 1
                if use_cnt[val] == 0:
                    killed.append(val)
        inc, dec = ComputeIncDec(op, killed)
        s, t = states.ComputeState(inc, dec)
        if s > budget:
            return SchedResult(valid=False)
        states.append(inc, dec)
        for val in killed:
            use_cnt.pop(val, None)
        for val in op.outputs:
            use_cnt[val] = len(val.uses)
    return SchedResult(valid=True, seq=list(seq.ops), states=states)

def scheduleGroupRpo(group, use_cnt: Dict[Any, int], budget: int) -> SchedResult:
    """按反向后序遍历调度组"""
    opSeq = []
    states = MemStateVec()
    for vert in group.Range():
        seq = AsSequence(vert)
        result = scheduleSequence(seq, use_cnt, budget - states.Latest())
        if not result.valid:
            return result
        opSeq.extend(result.seq)
        states.Extend(result.states)
    return SchedResult(valid=True, seq=opSeq, states=states)

class HierScheduler:
    """分层调度器"""
    def __init__(self, hier: 'HierGraph', budget: int, groupMemo: Dict[Any, SchedResult]):
        self.hier = hier
        self.budget = budget
        self.groupMemo = groupMemo

    def Schedule(self) -> List['OpRef']:
        predCnt = {vert: len(vert.preds) for vert in self.hier.RpoHierRange()}
        use_cnt = {}
        for input in self.hier.inputs:
            for succ in input.succs:
                predCnt[succ] -= 1
            use_cnt[input.value] = len(input.value.uses)
        zeroIn = []
        extractZeroIn(predCnt, zeroIn)
        initSize = sum(input.value.type.Size() for input in self.hier.inputs)
        memo = {tuple(zeroIn): PartialSchedResult(seq=[], states=MemStateVec(initSize), pred_cnt=predCnt, use_cnt=use_cnt)}
        for i in range(len(predCnt)):
            newMemo = {}
            for zeroInKey, result in memo.items():
                for vert in list(zeroInKey):
                    useCntCopy = result.use_cnt.copy()
                    vertResult = self.scheduleVertex(vert, useCntCopy, result.states)
                    self.updateResult(vert, list(zeroInKey), result, vertResult, useCntCopy, newMemo)
            memo = newMemo
        return memo[tuple()].seq

    def scheduleVertex(self, vert, use_cnt: Dict[Any, int], prevStates: 'MemStateVec') -> SchedResult:
        localBudget = self.budget - prevStates.Latest()
        if vert.Kind() == HierKind.SEQUENCE:
            return scheduleSequence(CastSequence(vert), use_cnt, localBudget)
        elif vert.Kind() == HierKind.GROUP:
            group = CastGroup(vert)
            ctx = GroupContext(group, use_cnt)
            if ctx in self.groupMemo:
                result = self.groupMemo[ctx]
                if result.states.Peak() > localBudget:
                    return SchedResult(valid=False)
                updateGroupUseCount(group, use_cnt)
                return result
            rpoUseCnt = use_cnt.copy()
            rpoBudget = min(localBudget, prevStates.Peak() - prevStates.Latest())
            rpoResult = scheduleGroupRpo(group, rpoUseCnt, rpoBudget)
            if rpoResult.valid:
                use_cnt.update(rpoUseCnt)
                return rpoResult
            dpResult = scheduleGroupDp(group, use_cnt, localBudget)
            if not dpResult.valid:
                return dpResult
            updateGroupUseCount(group, use_cnt)
            self.groupMemo[ctx] = dpResult
            return dpResult
        else:
            raise ValueError("Invalid vertex kind")

def updateGroupUseCount(group, use_cnt: Dict[Any, int]):
    """更新组的使用计数"""
    killed = []
    for val, num in group.consumed.items():
        use_cnt[val] -= num
        if use_cnt[val] == 0:
            killed.append(val)
    for val in killed:
        use_cnt.pop(val, None)
    use_cnt.update(group.produced)

def HierarchicalSchedule(graph: 'Graph') -> List['OpRef']:
    """分层调度主函数"""
    hier = HierGraph(graph)
    RunPass(hier, JoinSequencePass, MakeGroupPass)
    groupMemo = {}
    lastSched = []
    lastPeak = float('inf')
    while True:
        sched = HierScheduler(hier, lastPeak, groupMemo).Schedule()
        stat = ComputeLifetime(sched, graph)
        peak = EstimatePeak(sched, graph.inputs)
        peakValues = set()
        sizeRange = stat.SizeRange()
        for sizeKey, sizeVal in sizeRange.items():
            if sizeVal == peak:
                peakValues.update(sizeKey.AliveValues())
        if not peakValues:
            break
        if peak < lastPeak:
            lastPeak = peak
            lastSched = sched
        relSeqs = set()
        for val in peakValues:
            relSeqs.add(hier.opToSeq[val.def_.lock()])
        changed = False
        for seq in relSeqs:
            group = seq.group.lock()
            if group:
                ungroup(group)
                changed = True
            changed |= tryUngroupSucc(seq)
        if not changed:
            break
    return lastSched

def SerenitySchedule(graph: 'Graph', joinOps: bool, trySimple: bool, nSamples: int) -> List['OpRef']:
    """主调度函数，结合多种策略"""
    hier = HierGraph(graph)
    if joinOps:
        hier =  JoinSequencePass(hier)
    hier = MakeGroupPass(hier)
    topVerts = [v for v in hier.RpoHierRange()]
    sched = []
    states = MemStateVec()
    use_cnt = {}
    for i, vert in enumerate(topVerts):
        logger.info(f"Scheduling vertex {i+1}/{len(topVerts)}")
        if vert.Kind() == HierKind.INPUT:
            input = CastInput(vert)
            use_cnt[input.value] = len(input.value.uses)
            states = MemStateVec(input.value.type.Size())
        elif vert.Kind() == HierKind.SEQUENCE:
            result = scheduleSequence(CastSequence(vert), use_cnt, float('inf'))
            sched.extend(result.seq)
            states.Extend(result.states)
        elif vert.Kind() == HierKind.GROUP:
            group = CastGroup(vert)
            if trySimple:
                rpoUseCnt = use_cnt.copy()
                rpoResult = scheduleGroupRpo(group, rpoUseCnt, states.Peak() - states.Latest())
                if rpoResult.valid:
                    use_cnt = rpoUseCnt
                    sched.extend(rpoResult.seq)
                    states.Extend(rpoResult.states)
                    continue
            budget = float('inf')
            rng = random.Random()
            logger.info("Sampling schedules.")
            for _ in range(nSamples):
                budget = min(budget, sampleGroupPeak(group, use_cnt, rng))
            logger.info(f"Scheduling group with budget {budget / 1024} KB.")
            dpResult = scheduleGroupDp(group, use_cnt, budget)
            sched.extend(dpResult.seq)
            states.Extend(dpResult.states)
    return sched