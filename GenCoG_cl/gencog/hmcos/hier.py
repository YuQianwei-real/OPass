from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Set, TypeVar, Union, Generic

from ..graph.base import Vertex, Graph, Operation, Value, GraphVisitor, Input, Output


class HierKind(IntEnum):
    INPUT = auto()
    OUT = auto()
    SEQUENCE = auto()
    GROUP = auto() 
    

class HierVertex(Vertex):
    kind : HierKind

    def __init__(self):
        super().__init__()
        #Save the original predecessors of the vertices before grouping
        self.prePreds = self.preds 
        #Save the original successors of the vertices before grouping
        self.preSuccs = self.succs

        from .dom import DomNode
        self.dom: 'DomNode' = None
        self.postDom : 'DomNode'= None
    
    def Dominates(self, other: HierVertex, strict = False):
        return self.dom.Dominates(other.dom, strict)
    def PostDominates(self, other: HierVertex, strict = False):
        return self.postDom.Dominates(other.dom, strict)

class HierInput(HierVertex):
    kind = HierKind.INPUT

    def __init__(self, in_: Input):
        super().__init__()
        self.in_ = in_
        

class HierOutput(HierVertex):
    kind = HierKind.OUT

    def __init__(self, out_ :Output):
        super().__init__()
        self.out_ = out_

#操作序列与组
# A sequence of ops. All ops, except the first one, must only consume values produced by op in front of it.
# All ops, except the last one, must produce values that are all consumed by op next to it.
class Sequence(HierVertex):
    kind = HierKind.SEQUENCE

    def __init__(self, op: Operation):
        super().__init__()
        self.oprs_: List[Operation] = [op]
        self.inputs_ = op.inputs_  # List[Value]
        self.outputs_ = op.outputs_  # List[Value]
        self.group_ = None  # belong to Group

    def Label(self) -> str:
        return f"Seq({self.oprs_[0].op_.name_})"

    def Contains(self, op: Operation) -> bool:
        return op in self.oprs_

    def __lt__(self, other: Sequence):
        return len(self.oprs_) < len(other.oprs_)

class Group(HierVertex): #a group of sequences.  

    kind = HierKind.GROUP

    def __init__(self):
        super().__init__()
        self.seqs: List[Sequence] = []  # All sequences in this group
        self.entrs: List[Sequence] = []  # Entrance sequences of this group. Predecessors of each entrance must all be outside of the group.
        self.exits: List[Sequence] = []  # exit sequences of this group. Successors of each exits must all be outside of the group
        #In and out frontiers of the this group
        self.inFront: List[Sequence] = []  #Each input frontier must have at least one predecessor from sequence outside the group.
        self.outFront: List[Sequence] = []  # Each output frontier must have at least one successor from sequence outside the group
        #Use count of input and output values
        #Here we adopt producer-consumer model to describe def-use chains. When a value is defined, it produces a number of use counts. 
        #When it is used,  it consumes one use count. Only def-use chains across groups are counted.
        self.consumed: List[(Value, int)] = []  # List[(Value, int)]
        self.produced: List[(Value, int)]= []  # List[(Value, int)]

    def Label(self) -> str:
        return "Group"

    def Contains(self, item):
        if isinstance(item, Sequence):
            return item in self.seqs
        elif isinstance(item, Operation):
            return any(seq.Contains(item) for seq in self.seqs)
        return False

    def Range(self):

        return

#分层图结构
class HierGraph:
    def __init__(self, graph: Graph):
        # 原始计算图
        self.graph_ = graph
        #virtual root of hier_inputs
        self.root = HierVertex()

        vert_map: Dict[Vertex, HierVertex] = {} #vertex to hier_vertex
        # 将 op 映射到其所在的 Sequence
        self.op_to_seq: Dict[Operation, Sequence] = {}

        # 输入和输出节点
        self.hierIns_: List[HierInput]= []
        self.hierOuts_: List[HierOutput] = []
        allInputs = graph.inputs_ + graph.constants_ + graph.globals_
        for inp in allInputs:
            hierInp = HierInput(inp)
            self.hierIns_.append(hierInp)
            vert_map[inp] = hierInp
            #connect virtual_root and hier_inputs
            hierInp.preds.append(self.root)
            self.root.succs.append(hierInp)

        for out in graph.outputs_:
            hierOut =  HierOutput(out)
            self.hierOuts_.append(hierOut)
            vert_map[out] = hierOut
    

        # 初始化操作节点并创建对应的 Sequence
        self.sequences_: List[Sequence] = []
        for op in graph.oprs_:
            seq = Sequence(op)
            self.sequences_.append(seq)
            vert_map[op] = seq
            self.op_to_seq[op] = seq

        # 连接所有节点的前驱和后继
        for vert, hier in vert_map.items():
            assert isinstance(vert, Vertex) and isinstance(hier, HierVertex)
            for pred in vert.preds:
                hier.preds.append(vert_map[pred] if pred in vert_map else None)
            for succ in vert.succs:
                hier.succs.append(vert_map[succ] if succ in vert_map else None)
        

    def DfsHierRange(self):
        stack = [self.root]
        visited = set()
        hier_range = []
    
        while stack:
            vert = stack.pop()
            if vert not in visited:
                visited.add(vert)
                hier_range.append(vert)
            # 反向添加后继节点以保持正确DFS顺序
                for succ in reversed(vert.succs):
                    if succ not in visited:
                        stack.append(succ)
        
        return hier_range
        hier_range: List[HierVertex] = [self.root]
        visited: List[HierVertex] = []

        while len(visited) < len(hier_range):
            not_visited = [v for v in hier_range if v not in visited]
                
            for vert in not_visited:
                assert isinstance(vert, HierVertex)
                visited.append(vert)
                for succ in vert.succs:
                    if succ not in hier_range:
                        hier_range.append(succ)          

        assert len(hier_range) == len(visited), "failed to visit all hier_verts"
        return hier_range  
    
    def RpoHierRange(self):
        stack = [self.hierOuts_[0]]
        visited = set()
        reverse_dfs_order = []
    
        while stack:
            vert = stack.pop()
            if vert not in visited:
                visited.add(vert)
                reverse_dfs_order.append(vert)
            
                # 注意：这里是遍历前驱节点（preds）而不是后继节点（succs）
                # 并且反向添加以保持正确的DFS顺序
                for pred in reversed(vert.preds):
                    if pred not in visited:
                        stack.append(pred)
        reverse_dfs_order.remove(self.root)
        return reverse_dfs_order
        

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
        up = self.Latest() + inc
        down = up - dec
        return up, down

    def Append(self, inc, dec):
        up, down = self.Compute_state(inc, dec)
        self.stables.append(up)
        self.transients.append(down)

    def Extend(self, other: MemStateVec):
        latest = self.Latest()
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


R = TypeVar('R')

#迭代器与访问者模式
class HierGraphBuilder(GraphVisitor[None]):
    def __init__(self):
        self.memo: Dict[Vertex, Any] = {}

    def visit(self, vert: Vertex, *args, **kwargs) -> Any:
        if vert in self.memo:
            return self.memo[vert]

        kind = vert.Kind()
        method_name = {
            'input': 'visit_input',
            'output': 'visit_output',
            'sequence': 'visit_sequence',
            'group': 'visit_group'
        }.get(kind)

        if not method_name:
            raise ValueError(f"Unsupported vertex kind: {kind}")

        method = getattr(self, method_name, None)
        if not method:
            raise NotImplementedError(f"{method_name} not implemented")

        result = method(vert, *args, **kwargs)
        self.memo[vert] = result
        return result

    @abstractmethod
    def visit_input(self, input_vert: HierInput, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def visit_output(self, output_vert: HierOutput, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def visit_sequence(self, sequence: Sequence, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def visit_group(self, group: Group, *args, **kwargs) -> Any:
        pass

#Pass 接口与执行
class HierGraphPass:
    def run(self, graph: HierGraph):
        raise NotImplementedError

def run_pass(graph: HierGraph, *passes):
    for p in passes:
        p.run(graph)