from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
import logging
import copy
import abc

# 假设已定义如下类和类型别名
from HMOCS.core.value import Value, ValueRef, ValueKind
from HMCOS.util.util import LOG_FATAL
from HMCOS.core.vertex import VertexBase

# 定义类型别名
VertexRef = 'Vertex'
InputRef = 'Input'
OutputRef = 'Output'
OpRef = 'Op'


class VertexKind(IntEnum):
    INPUT = auto()
    OUTPUT = auto()
    OP = auto()


class Vertex(VertexBase):
    """
    Base class of computation graph vertex.
    """
    def kind(self) -> VertexKind:
        raise NotImplementedError


class Input(Vertex):
    """
    Input placeholder of computation graph.
    """
    def __init__(self, value: ValueRef):
        self.value = value
        assert value.kind == ValueKind.INPUT, "Value must be of kind INPUT"

    @property
    def class_kind(self) -> VertexKind:
        return VertexKind.INPUT

    def kind(self) -> VertexKind:
        return VertexKind.INPUT

class Output(Vertex):
    def __init__(self, value: ValueRef):
        self.value = value
        assert value.kind == ValueKind.RESULT, "Value must be of kind RESULT"

    def def_(self) -> VertexRef:
        return self.preds[0]  # 假设 preds 是 VertexRef 列表

    @property
    def class_kind(self) -> VertexKind:
        return VertexKind.OUTPUT

    def kind(self) -> VertexKind:
        return VertexKind.OUTPUT

class Op(Vertex):
    def __init__(self, node: 'onnx.NodeProto'):
        self.name = node.name()
        self.type = node.op_type()
        self.inputs: List[ValueRef] = []
        self.outputs: List[ValueRef] = []

    def __init__(self, other: 'Op'):
        self.name = other.name
        self.type = other.type
        self.inputs = copy.copy(other.inputs)
        self.outputs = copy.copy(other.outputs)

    @property
    def class_kind(self) -> VertexKind:
        return VertexKind.OP

    def kind(self) -> VertexKind:
        return VertexKind.OP


class Graph:
    def __init__(self, model: Optional['onnx.ModelProto'] = None, name: str = ""):
        self.name = name
        self.inputs: List[InputRef] = []
        self.outputs: List[OutputRef] = []
        self.params: List[ValueRef] = []
        self.ops: List[OpRef] = []

        if model is not None:
            # 构造逻辑，假设 model 中有 value_info 信息
            pass

    def clone(self) -> 'Graph':
        # 深拷贝逻辑
        new_graph = Graph(name=self.name)
        new_graph.inputs = [copy.deepcopy(i) for i in self.inputs]
        new_graph.outputs = [copy.deepcopy(o) for o in self.outputs]
        new_graph.params = [copy.deepcopy(p) for p in self.params]
        new_graph.ops = [copy.deepcopy(op) for op in self.ops]
        return new_graph

    def subgraph(self, is_output: Callable[[OpRef], bool], sub_name: str) -> 'Graph':
        # 提取子图逻辑
        pass

    def plot(self, directory: str, format: str = "pdf"):
        # 可视化逻辑
        pass

    def connect_verts(self):
        # 连接顶点逻辑
        pass


#迭代器类
class VertRange:
    def __init__(self, vertices: List[VertexRef]):
        self.vertices = vertices

class RpoVertRange(VertRange):
    def __init__(self, graph: Graph):
        super().__init__([VertexRef(out) for out in graph.outputs])

class DfsVertRange(VertRange):
    def __init__(self, graph: Graph):
        super().__init__([VertexRef(inp) for inp in graph.inputs])


#访问者模式
class VertexVisitor(ABC):
    def __init__(self):
        self.memo: Dict[VertexRef, Any] = {}

    def visit(self, vert: VertexRef, *args, **kwargs):
        if vert in self.memo:
            return self.memo[vert]
        ret = None
        if isinstance(vert, Input):
            ret = self.visit_input(vert, *args, **kwargs)
        elif isinstance(vert, Output):
            ret = self.visit_output(vert, *args, **kwargs)
        elif isinstance(vert, Op):
            ret = self.visit_op(vert, *args, **kwargs)
        else:
            LOG_FATAL("Unreachable")
        self.memo[vert] = ret
        return ret

    @abstractmethod
    def visit_input(self, input: InputRef, *args, **kwargs):
        pass

    @abstractmethod
    def visit_output(self, output: OutputRef, *args, **kwargs):
        pass

    @abstractmethod
    def visit_op(self, op: OpRef, *args, **kwargs):
        pass


#克隆器实现
class VertexCloner(VertexVisitor):
    def __init__(self):
        super().__init__()
        self.value_map: Dict[ValueRef, ValueRef] = {}

    def visit_input(self, input: InputRef) -> VertexRef:
        new_value = self.visit_value(input.value)
        return Input(new_value)

    def visit_output(self, output: OutputRef) -> VertexRef:
        new_value = self.visit_value(output.value)
        return Output(new_value)

    def visit_op(self, op: OpRef) -> VertexRef:
        # 克隆 Op 并处理 inputs/outputs
        new_op = Op(op)
        new_op.inputs = [self.visit_value(v) for v in op.inputs]
        new_op.outputs = [self.visit_value(v) for v in op.outputs]
        return new_op

    def visit_value(self, value: ValueRef) -> ValueRef:
        if value in self.value_map:
            return self.value_map[value]
        new_value = copy.deepcopy(value)
        self.value_map[value] = new_value
        return new_value