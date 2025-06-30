import enum
import math
import weakref
from typing import List, Optional, Union, Any
from onnx import TensorProto, TypeProto

from HMCOS.core.graph import Op, Input, Vertex


# ======================
# Enum Definitions
# ======================

class DataType(enum.Enum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16


class ValueKind(enum.Enum):
    INPUT = 0
    PARAM = 1
    RESULT = 2


# ======================
# TensorType Class
# ======================

class TensorType:
    def __init__(self, shape: List[int], dtype: DataType):
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def from_tensor(tensor: TensorProto) -> 'TensorType':
        shape = list(tensor.dims)
        dtype = DataType(tensor.data_type)
        return TensorType(shape, dtype)

    @staticmethod
    def from_type(type_proto: TypeProto.Tensor) -> 'TensorType':
        shape = [dim.dim_value for dim in type_proto.shape.dim]
        dtype = DataType(type_proto.elem_type)
        return TensorType(shape, dtype)

    def count(self) -> int:
        return math.prod(self.shape)

    def size(self) -> int:
        dtype_size = {
            DataType.FLOAT: 4,
            DataType.UINT8: 1,
            DataType.INT8: 1,
            DataType.UINT16: 2,
            DataType.INT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.FLOAT16: 2,
            DataType.DOUBLE: 8,
            DataType.UINT32: 4,
            DataType.UINT64: 8,
            DataType.COMPLEX64: 8,
            DataType.COMPLEX128: 16,
            DataType.BFLOAT16: 2,
            # STRING and BOOL are variable-length types
        }.get(self.dtype, 0)
        return self.count() * dtype_size

    def __eq__(self, other: 'TensorType') -> bool:
        return self.shape == other.shape and self.dtype == other.dtype


# ======================
# Value Class
# ======================

class Value:
    def __init__(
        self,
        kind: ValueKind,
        name: str,
        type_: TensorType
    ):
        self.kind = kind
        self.name = name
        self.type = type_
        self.input: Optional[weakref.ReferenceType['Input']] = None
        self.data: bytes = b''
        self.def_: Optional[weakref.ReferenceType['Op']] = None
        self.uses: List[weakref.ReferenceType['Op']] = []

    @classmethod
    def create_input(cls, info: TypeProto.Tensor) -> 'Value':
        tensor_type = TensorType.from_type(info)
        return cls(kind=ValueKind.INPUT, name=info.name, type_=tensor_type)

    @classmethod
    def create_param(cls, tensor: TensorProto) -> 'Value':
        tensor_type = TensorType.from_tensor(tensor)
        value = cls(kind=ValueKind.PARAM, name=tensor.name, type_=tensor_type)
        value.data = tensor.raw_data
        return value

    @classmethod
    def create_result(cls, info: TypeProto.Tensor) -> 'Value':
        tensor_type = TensorType.from_type(info)
        return cls(kind=ValueKind.RESULT, name=info.name, type_=tensor_type)

    def __copy__(self) -> 'Value':
        new_value = Value(kind=self.kind, name=self.name, type_=self.type)
        new_value.data = self.data  # shallow copy
        return new_value

    def vertex(self) -> Optional['Vertex']:
        if self.kind == ValueKind.INPUT:
            if self.input is not None:
                return self.input()
            return None
        elif self.kind == ValueKind.RESULT:
            if self.def_ is not None:
                return self.def_().vertex  # 假设 Op 有一个 vertex 属性
            return None
        else:
            return None  # 参数没有顶点