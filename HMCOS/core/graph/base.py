from enum import IntEnum, auto
from typing import List, TypeVar, Generic, Dict, Callable, Any, Optional, Tuple

from ..expr.ty import ValueType, TypeCode
from ..solve import TensorType, RelayType, VarTensorType
from ..spec import Op
#from .vertex import VertexBase



class VertexKind(IntEnum):
    IN = auto()
    OUT = auto()
    OPR = auto()
    GV = auto()
    CON = auto()

class Vertex():
    """
    Base class of computation graph vertex.
    """
    kind: VertexKind

    def __init__(self):
        self.preds: List[Vertex] = []
        self.succs: List[Vertex] = []
    


class Input(Vertex):
    """
    Input placeholder of computation graph.
    """
    kind = VertexKind.IN

    def __init__(self, ty: TensorType, is_param: bool):
        self.value_ = Value(ty, def_vert=self)
        self.is_param_ = is_param

    @classmethod
    def from_value(cls, value: 'Value', is_param: bool):
        obj = cls(value.type_, is_param)
        obj.value_ = value
        value.def_ = obj
        return obj

class Constant(Vertex):
    """
    Constant of computation graph.
    """
    kind = VertexKind.CON

    def __init__(self, ty: TensorType, data: ValueType):
        self.value_ = Value(ty, def_vert=self)
        self.data_ = data

    @property
    def data_repr(self):
        assert self.value_.type_.rank == 0, str(self.value_.type_)
        if self.value_.type_.dtype_.code_ in (TypeCode.int, TypeCode.uint):
            return str(self.data_)
        else:
            return str(self.data_) + 'f'

class Global(Vertex):
    """
    Global variable of compuation graph'
    """
    kind = VertexKind.GV

    def __init__(self, ty: RelayType, name: str) -> None:
        self.value_ = Value(ty, def_vert=self)
        self.name_ = name

class Output(Vertex):
    """
    Output indicator of computation graph.
    """
    kind = VertexKind.OUT

    def __init__(self, value: 'Value'):
        self.value_ = value
        value.uses_.append(self)


class Operation(Vertex):
    """
    Call of operator on tensor values.
    """
    kind = VertexKind.OPR

    def __init__(self, op: Op, attrs: List[Tuple[str, ValueType]],
                 inputs: List['Value'], outputs: List['Value']):
        self.op_ = op
        self.attrs_ = attrs
        self.inputs_ = inputs
        for i in self.inputs_:
            i.uses_.append(self)
        self.outputs_ = outputs
        for o in self.outputs_:
            o.def_ = self


class Value:
    """
    Tensor value defined by vertex.
    """

    def __init__(self, ty: TensorType, def_vert: Optional[Vertex] = None):
        self.type_ = ty
        self.def_ = def_vert
        self.uses_: List[Vertex] = []


class Graph:
    """
    Computation graph.
    """

    def __init__(self, ins: List[Input], outs: List[Output], oprs_: List[Operation], 
                 outrefs: List[Value] = [], typevars: List[VarTensorType] = []):
        self.inputs_ = ins
        self.outputs_ = outs
        self.oprs_ = oprs_
        self.outrefs_ = outrefs
        self.typevars_ = typevars

class GraphMod:
    """
    A graph group of a relay mod.
    """

    def __init__(self, graphs: Dict[str, Graph] = {}) -> None:
        self.graphs_ = graphs
    
    def __getitem__(self, fn: str) -> Graph:
        return self.graphs_[fn]
    
    def update(self, fn: str, g: Graph):
        self.graphs_[fn] = g

    @property
    def funcNames(self):
        return list(self.graphs_.keys())



R = TypeVar('R')


class GraphVisitor(Generic[R]):
    def __init__(self):
        self._methods: Dict[VertexKind, Callable[[Any], R]] = {
            VertexKind.IN: self.visit_input,
            VertexKind.OUT: self.visit_output,
            VertexKind.OPR: self.visit_operation,
            VertexKind.GV: self.visit_global,
            VertexKind.CON: self.visit_constant, 
        }
        self._vert_memo: Dict[Vertex, R] = {}

    def visit(self, v: Vertex):
        if v in self._vert_memo:
            return self._vert_memo[v]
        r = self._methods[v.kind](v)
        self._vert_memo[v] = r
        return r

    def visit_input(self, i: Input) -> R:
        pass

    def visit_global(self, g: Global) -> R:
        pass

    def visit_constant(self, c: Constant) -> R:
        pass

    def visit_output(self, o: Output) -> R:
        pass

    def visit_operation(self, opr: Operation) -> R:
        pass

    def visit_value(self, v: Value):
        pass

# class GraphModVisitor(GraphVisitor):
#     def __init__(self) -> None:
#         super().__init__()
#         self._methods[Graph]