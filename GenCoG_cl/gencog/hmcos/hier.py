from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
import weakref
from __future__ import annotations

from ..graph.base import Vertex, Graph, Operation, Value, VertexKind


class HierKind(IntEnum):
    SQ = auto()
    GP = auto() 

class HierVertex:

    kind : HierKind

    def __init__(self, v: Vertex):
        self.Preds: List[HierVertex] = v.preds
        self.Succs: List[HierVertex] = v.succs


#操作序列与组

class Sequence(HierVertex):

    kind = HierKind.SQ

    def __init__(self, op: Operation):
        super().__init__()
        self.oprs_: List[Operation] = [op]
        self.Preds: List[HierVertex] = op.preds
        self.Succs: List[HierVertex] = op.succs
        self.inputs_ = op.inputs_  # List[Value]
        self.outputs_ = op.outputs_  # List[Value]
        self.group = None  # weakref to Group

    def Label(self) -> str:
        return f"Seq({self.oprs_[0].name})"

    def Contains(self, op: Operation) -> bool:
        return op in self.oprs_


class Group(Vertex):

    kind = VertexKind.GP

    def __init__(self):
        super().__init__()
        self.seqs = []  # List[Sequence]
        self.entrs = []  # List[Sequence]
        self.exits = []  # List[Sequence]
        self.inFront = []  # List[Sequence]
        self.outFront = []  # List[Sequence]
        self.consumed = []  # List[(ValueRef, int)]
        self.produced = []  # List[(ValueRef, int)]

    def Label(self) -> str:
        return "Group"


    def Contains(self, item):
        if isinstance(item, Sequence):
            return item in self.seqs
        elif isinstance(item, Operation):
            return any(seq.Contains(item) for seq in self.seqs)
        return False


class MemStateVec:
    def __init__(self, init=0):
        self.init = init
        self.stables = []  # 稳定状态列表
        self.transients = []  # 瞬时状态列表

    def Latest(self):
        return self.init if not self.transients else self.transients[-1]

    def Peak(self):
        return self.init if not self.stables else max(self.stables)

    def Compute_state(self, inc, dec):
        up = self.latest() + inc
        down = up - dec
        return (up, down)

    def Append(self, inc, dec):
        up, down = self.compute_state(inc, dec)
        self.stables.append(up)
        self.transients.append(down)

    def Extend(self, other: MemStateVec):
        latest = self.latest()
        for i in range(len(other.stables)):
            self.stables.append(other.stables[i] + latest)
            self.transients.append(other.transients[i] + latest)

    def Swap(self, other: MemStateVec):
        self.init, other.init = other.init, self.init
        self.stables, other.stables = other.stables, self.stables
        self.transients, other.transients = other.transients, self.transients

    def __getitem__(self, i):
        assert i < len(self.stables), "Index out of range"
        return (self.stables[i], self.transients[i])

    def Size(self):
        return len(self.stables)

    def Get_stables(self):
        return self.stables

    def Get_transients(self):
        return self.transients


class SchedResult:
    def __init__(self, seq: List[Operation], states: MemStateVec):
        self.seq = seq  # 调度的操作序列
        self.states = states  # 内存状态
        self.valid = True  # 标记是否有效（未超预算）



#分层图结构

class HierGraph:
    def __init__(self, graph: Graph):
        self.graph_ = graph
        self.inputs_ = graph.inputs_
        self.outputs_ = graph.outputs_
        self.op_to_seq = {}  # Dict[OpRef, Sequence]

    def plot_all(self, directory: str, name: str, format: str = "pdf"):
        pass

    def plot_top(self, directory: str, name: str, format: str = "pdf"):
        pass

    def plot_dom(self, directory: str, name: str, format: str = "pdf"):
        pass

    def plot_post_dom(self, directory: str, name: str, format: str = "pdf"):
        pass


#迭代器与访问者模式




#Pass 接口与执行
class HierGraphPass:
    def run(self, graph: HierGraph):
        raise NotImplementedError

def run_pass(graph: HierGraph, *passes):
    for p in passes:
        p.run(graph)