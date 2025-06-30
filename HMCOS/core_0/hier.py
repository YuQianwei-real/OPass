from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
import weakref

from HMCOS.core.vertex import VertexBase

class HierKind(IntEnum):
    INPUT = auto()
    OUTPUT = auto()
    SEQUENCE = auto()
    GROUP = auto()



class HierVertex(VertexBase):
    def __init__(self):
        self.dom = None  # DomNode<HierVertex>
        self.postDom = None  # DomNode<HierVertex>
        self.prevPreds = []  # List of weakref to HierVertex
        self.prevSuccs = []  # List of HierVertex

    def Label(self) -> str:
        raise NotImplementedError

    def Kind(self) -> HierKind:
        raise NotImplementedError

    def Dominates(self, other: 'HierVertex', strict: bool = False) -> bool:
        if not self.dom:
            return False
        return self.dom.Dominates(other.dom, strict)

    def PostDominates(self, other: 'HierVertex', strict: bool = False) -> bool:
        if not self.postDom:
            return False
        return self.postDom.Dominates(other.postDom, strict)

    def BackupEdges(self):
        self.prevPreds = [weakref.ref(p) for p in self.prevPreds]
        self.prevSuccs = list(self.prevSuccs)



#输入输出节点
class ValueRef:
    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind  # INPUT, RESULT, etc.

class HierInput(HierVertex):
    def __init__(self, value: ValueRef):
        super().__init__()
        assert value.kind == 'INPUT', "Value must be of kind INPUT"
        self.value = value

    def Label(self) -> str:
        return self.value.name

    def Kind(self) -> HierKind:
        return HierKind.INPUT

class HierOutput(HierVertex):
    def __init__(self, value: ValueRef):
        super().__init__()
        assert value.kind == 'RESULT', "Value must be of kind RESULT"
        self.value = value

    def Label(self) -> str:
        return self.value.name

    def Kind(self) -> HierKind:
        return HierKind.OUTPUT
    

#操作序列与组
class OpRef:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

class Sequence(HierVertex):
    def __init__(self, op: OpRef):
        super().__init__()
        self.ops = [op]
        self.inputs = []  # List[ValueRef]
        self.outputs = []  # List[ValueRef]
        self.group = None  # weakref to Group

    def Label(self) -> str:
        return f"Seq({self.ops[0].name})"

    def Kind(self) -> HierKind:
        return HierKind.SEQUENCE

    def Contains(self, op: OpRef) -> bool:
        return op in self.ops

class Group(HierVertex):
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

    def Kind(self) -> HierKind:
        return HierKind.GROUP

    def Contains(self, item):
        if isinstance(item, Sequence):
            return item in self.seqs
        elif isinstance(item, OpRef):
            return any(seq.Contains(item) for seq in self.seqs)
        return False


#分层图结构
class Graph:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.ops = []

class HierGraph:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.inputs = []  # List[HierInput]
        self.outputs = []  # List[HierOutput]
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
class VertRange:
    def __init__(self, vertices):
        self.vertices = vertices

class RpoHierRange(VertRange):
    def __init__(self, hier_graph):
        super().__init__([v for v in hier_graph.outputs])

class HierVertVisitor(ABC):
    def __init__(self):
        self.memo = {}

    def visit(self, vert, *args, **kwargs):
        if vert in self.memo:
            return self.memo[vert]
        ret = None
        kind = vert.Kind()
        if kind == HierKind.INPUT:
            ret = self.visit_input(vert, *args, **kwargs)
        elif kind == HierKind.OUTPUT:
            ret = self.visit_output(vert, *args, **kwargs)
        elif kind == HierKind.SEQUENCE:
            ret = self.visit_sequence(vert, *args, **kwargs)
        elif kind == HierKind.GROUP:
            ret = self.visit_group(vert, *args, **kwargs)
        else:
            raise RuntimeError("Unreachable")
        self.memo[vert] = ret
        return ret

    @abstractmethod
    def visit_input(self, input: HierInput, *args, **kwargs):
        pass

    @abstractmethod
    def visit_output(self, output: HierOutput, *args, **kwargs):
        pass

    @abstractmethod
    def visit_sequence(self, seq: Sequence, *args, **kwargs):
        pass

    @abstractmethod
    def visit_group(self, group: Group, *args, **kwargs):
        pass



#Pass 接口与执行
class HierGraphPass:
    def run(self, graph: HierGraph):
        raise NotImplementedError

def run_pass(graph: HierGraph, *passes):
    for p in passes:
        p.run(graph)