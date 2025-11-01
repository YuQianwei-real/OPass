from typing import Dict, List, cast, Iterable, Any

import numpy as np
from tvm import relay, tir, runtime, ir
from tvm.ir import IRModule

from .base import GraphVisitor, GraphMod, Value, Graph, VertexKind, Input, Output, Operation, Global, Constant
from ..solve import TensorType, RelayType, RelayTypeKind, TupleTensorType, RefTensorType, VarTensorType, FuncTensorType
from ..expr.ty import ValueType, DataType
from ..spec import OpRegistry
from ..util import NameGenerator, CodeBuffer

# Operators that accept tuple as input
tuple_in_ops = {
    'concatenate',
}

# Operators that return a tuple, no matter how many output values they produce
tuple_out_ops = {
    'split',
}

# Virtual operators
virtual_ops = [
    'let',
    'tuple',
    'getitem',
    'func',
    'def',
    'call',
]

def print_relay(graphs: GraphMod):
    return RelayPrinter().print(graphs)

def fmt_val(v: ValueType):
    if isinstance(v, (bool, int, DataType)):
        return str(v)
    elif isinstance(v, float):
        return str(v) + 'f'
    elif isinstance(v, str):
        return '"' + v + '"'
    elif isinstance(v, (tuple, list)):
        return '[' + ', '.join(fmt_val(e) for e in v) + ']'
    elif v is None:
        # return fmt_val([])
        return 'None'
    else:
        assert False, type(v)


class RelayPrinter(GraphVisitor[None]):
    def __init__(self):
        super().__init__()
        self._buf = CodeBuffer()
        self._val_names: Dict[Value, str] = {}
        self._arg_gen = NameGenerator('%x')
        self._res_gen = NameGenerator('%')

    def print(self, gmod: GraphMod):
        self._buf.writeln('#[version = "0.0.5"]')
        for fn in gmod.funcNames:
            self.print_func(gmod[fn], fn)
        return str(self._buf)

    def print_func(self, g: Graph, fn: str):
        # Function signature
        typevars = [v.name_ for v in g.typevars_]
        typevar_str = ', '.join(typevars)
        self._buf.write(f'def @{fn}')
        if len(typevars) > 0:
            self._buf.write(f'[{typevar_str}]')
        inputs = [f'{self.visit_value(i.value_)}: {i.value_.type_}' if str(i.value_.type_) != 'ref' else f'{self.visit_value(i.value_)}' for i in g.inputs_]
        inputs_str = ', '.join(inputs)
        self._buf.write(f'({inputs_str})')
        # self._buf.write_pos(
        #     map(lambda i: lambda: self._buf.write(
        #         f'{self.visit_value(i.value_)}: {i.value_.type_}'), g.inputs_)
        # )
        # self._buf.write(' -> ')
        # assert len(g.outputs_) == 1
        # self._buf.write(str(g.outputs_[0].value_.type_))

        # Function body
        self._buf.writeln(' {')
        with self._buf.indent():
            for outref in g.outrefs_:
                self.visit(outref.def_)

            for out in g.outputs_:
                self.visit(out)

            # Print unvisited operators.
            for opr in g.oprs_:
                if opr not in self._vert_memo:
                    self.visit(opr)

            out_str = self.visit_value(g.outputs_[0].value_)
            self._buf.writeln(out_str)
        self._buf.writeln('}')

    def visit_input(self, i: Input):
        return

    def visit_output(self, o: Output):
        return self.visit(o.value_.def_)

    def visit_operation(self, opr: Operation):
        # Check op type
        op_name = opr.op_.name_
        
        # Visit predecessors
        if op_name != 'def':
            for v in opr.inputs_:
                self.visit(v.def_)

        # Common operators
        if op_name not in virtual_ops:
            # Print output value
            assert len(opr.outputs_) == 1
            self._buf.write(self.visit_value(opr.outputs_[0]))
            self._buf.write(' = ')

            # Print operator call
            self._buf.write(opr.op_.name_)
            write_task = []
            if len(opr.inputs_) > 0:
                args = map(lambda v: self.visit_value(v), opr.inputs_)
                arg_str = ', '.join(args)
                write_task.append(lambda: self._buf.write(arg_str))
            if len(opr.attrs_) > 0:
                write_task.append(lambda: self._buf.write_named(
                    map(lambda a: (a[0], lambda: self._buf.write(fmt_val(a[1]))), opr.attrs_),
                    prefix='', suffix=''
                ))
            self._buf.write_pos(write_task)
            self._buf.writeln(f';')
            # ty_str = repr(opr.outputs_[0].type_)
            # self._buf.writeln(f'; /* ty={ty_str} */')
        
        elif op_name == 'let':
            assert len(opr.outputs_) == 1 and len(opr.inputs_) == 1
            self._buf.writeln(f'let {self.visit_value(opr.outputs_[0])} = {self.visit_value(opr.inputs_[0])};')
        
        elif op_name == 'tuple':
            assert len(opr.outputs_) == 1
            arg_str = ', '.join([self.visit_value(inp) for inp in opr.inputs_])
            if len(opr.inputs_) == 1:
                arg_str += ','
            self._buf.writeln(f'{self.visit_value(opr.outputs_[0])} = ({arg_str});')

        elif op_name == 'getitem':
            assert len(opr.outputs_) == 1 and len(opr.inputs_) == 1
            self._buf.writeln(f'{self.visit_value(opr.outputs_[0])} = {self.visit_value(opr.inputs_[0])}.{opr.attrs_[0][1]};')

        elif op_name == 'call':
            assert len(opr.outputs_) == 1 and len(opr.inputs_) > 1
            # Print output value
            self._buf.write(self.visit_value(opr.outputs_[0]))
            self._buf.write(' = ')

            # Print operator call
            self._buf.write(self.visit_value(opr.inputs_[0]))
            write_task = []
            if len(opr.inputs_) > 1:
                args = map(lambda v: self.visit_value(v), opr.inputs_[1:])
                arg_str = ', '.join(args)
                write_task.append(lambda: self._buf.write(arg_str))
            if len(opr.attrs_) > 0:
                write_task.append(lambda: self._buf.write_named(
                    map(lambda a: (a[0], lambda: self._buf.write(fmt_val(a[1]))), opr.attrs_),
                    prefix='', suffix=''
                ))
            self._buf.write_pos(write_task)
            self._buf.writeln(f';')

        elif op_name == 'def':
            assert len(opr.outputs_) == 1 and len(opr.inputs_) > 1
            # Print output value
            self._buf.write(self.visit_value(opr.outputs_[0]))
            self._buf.write(' = ')

            with self._buf.indent():
                # Visit function body
                self.visit(opr.inputs_[0].def_)

                # Print function output
                self.visit(opr.inputs_[-1].def_)
                self._buf.writeln(f'{self.visit_value(opr.inputs_[-1])}')

            # Print function end
            self._buf.writeln('};')       # Note ';' need to be processed

        elif op_name == 'func':
            assert len(opr.outputs_) > 0 and len(opr.inputs_) == 0
            # Print function head
            arg_str = ', '.join([self.visit_value(arg) for arg in opr.outputs_])
            self._buf.writeln(f'fn ({arg_str}) ' + '{')

    def visit_value(self, v: Value):
        if v in self._val_names:
            return self._val_names[v]
        if v.def_.kind == VertexKind.IN:
            name = self._arg_gen.generate()
        elif v.def_.kind == VertexKind.OPR and cast(Operation, v.def_).op_.name_ == 'let':
            name = self._arg_gen.generate()
        elif v.def_.kind == VertexKind.OPR and cast(Operation, v.def_).op_.name_ == 'func':
            name = self._arg_gen.generate()
        elif v.def_.kind == VertexKind.CON:
            name = str(cast(Constant, v.def_).data_repr)
        elif v.def_.kind == VertexKind.GV:
            name = '@' + str(cast(Global, v.def_).name_)
        else:
            name = self._res_gen.generate()
        self._val_names[v] = name
        return name

def build_graph(mod: IRModule) -> GraphMod:
    graphs: Dict[str, Graph] = {}
    for fn in mod.functions:
        func = mod[fn.name_hint]
        params = [param.name_hint for param in func.params]
        graphs[fn.name_hint] = build_func_graph(func, params)
    return GraphMod(graphs)

def build_func_graph(func: relay.function.Function, params: List[str]) -> Graph:
    return GraphBuilder(params).build(func)

class GraphBuilder(relay.ExprFunctor):
    def __init__(self, params: List[str]):
        super().__init__()
        self._params = params
        self._name2val: Dict[str, Value] = {}
        self._inputs: List[Input] = []
        self._globals: List[Global] = []
        self._constants: List[Constant] = []
        self._oprs: List[Operation] = []
        self._outrefs: List[Value] = []     # references after a series of write insts.
        self._typevars: Dict[str: VarTensorType] = {}

    def build(self, fn: relay.function.Function):
        # Create inputs
        self._inputs = [Input(self._cvt_type(var.checked_type), var.name_hint in self._params)
                        for var in fn.params]
        self._name2val = {p.name_hint: inp.value_ for p, inp in zip(fn.params, self._inputs)}

        # Create outputs
        outputs = [Output(self.visit(fn.body))]
        
        for out in outputs:
            out.value_.def_.succs.append(out)
            out.preds.append(out.value_.def_)

        # Create graph
        return Graph(self._inputs, outputs, self._oprs, outrefs=self._outrefs, typevars=list(self._typevars.values()), 
                     constants = self._constants, globals = self._globals)

    def visit_function(self, fn: relay.function.Function):
        # Create funcDef node along with params generated by funcDef
        param_values = [Value(self._cvt_type(var.checked_type)) for var in fn.params]
        self._name2val.update({p.name_hint: inp for p, inp in zip(fn.params, param_values)})
        opr = Operation(OpRegistry.get('func'), [], [], param_values)
        self._oprs.append(opr)

        # Visit function body   TODO: Check
        ret = [self.visit(fn.body)]

        # Create funcDef node
        funcValue = Value(self._cvt_type(fn.checked_type))
        return_opr = Operation(OpRegistry.get('def'), [], param_values + ret, [funcValue])
        self._oprs.append(return_opr)

        # 建立前驱-后继关系
        for p in param_values:
            if hasattr(p, 'def_') and p.def_ is not None:
                return_opr.preds.append(p.def_)
                p.def_.succs.append(return_opr)

        funcValue.def_ = return_opr
        
        return funcValue

    def visit_var(self, var: relay.Var):
        return self._name2val[var.name_hint]

    def visit_constant(self, const: relay.expr.Constant):
        inp = Constant(self._cvt_type(const.checked_type), data=self._cvt_ir_value(const.data))
        self._constants.append(inp)
        return inp.value_

    def visit_tuple(self, tup: relay.expr.Tuple):
        #Deal with every Fields
        inputs = [self.visit(f) for f in tup.fields]
        outputs = Value(self._cvt_type(tup.checked_type))

        #Create Operation
        opr = Operation(OpRegistry.get('tuple'), [], inputs, [outputs])
        self._oprs.append(opr)

        # 建立前驱-后继关系
        for inp in inputs:
            if hasattr(inp, 'def_') and inp.def_ is not None:
                opr.preds.append(inp.def_)
                inp.def_.succs.append(opr)

        outputs.def_ = opr

        return outputs

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        tup = self.visit(getitem.tuple_value)
        outputs = Value(self._cvt_type(getitem.checked_type))
        opr = Operation(OpRegistry.get('getitem'), [('index', getitem.index)], [tup], [outputs])
        self._oprs.append(opr)

        # 建立前驱-后继关系
        if hasattr(tup, 'def_') and tup.def_ is not None:
            opr.preds.append(tup.def_)
            tup.def_.succs.append(opr)
        
        outputs.def_ = opr

        return outputs

    def visit_call(self, call: relay.Call):
        # Collect input values
        if isinstance(call.op, (relay.RefRead, relay.Function, relay.Var, relay.Call, relay.GlobalVar)):
            name = 'call'
            fn = self.visit(call.op)
            inputs = [fn] + [self.visit(a) for a in call.args]
        else:
            name = call.op.name
            inputs = [self.visit(a) for a in call.args]

        # Convert attribute values
        if call.attrs is None or (not hasattr(call.attrs, 'keys')):
            attrs = []
        else:
            attrs = [(str(k), self._cvt_ir_value(call.attrs[k])) for k in call.attrs.keys()]


        # Create output values
        out_ty = call.checked_type
        outputs = Value(self._cvt_type(out_ty))

        # Create operation
        opr = Operation(OpRegistry.get(name), attrs, inputs, [outputs])
        self._oprs.append(opr)

        # 建立前驱-后继关系
        for inp in inputs:
            if hasattr(inp, 'def_') and inp.def_ is not None:
                opr.preds.append(inp.def_)  # 输入的定义操作是当前操作的前驱
                inp.def_.succs.append(opr)  # 当前操作是输入定义操作的后继

        outputs.def_ = opr  # 输出值的定义操作是当前操作

        return outputs

    def visit_let(self, let: relay.expr.Let):
        # Collect input values
        inputs = [self.visit(let.value)]

        # Create output values
        out_ty = let.var.checked_type
        outputs = Value(self._cvt_type(out_ty))
        self._name2val[let.var.name_hint] = outputs
        
        # Create operation
        opr = Operation(OpRegistry.get('let'), [], inputs, [outputs])
        self._oprs.append(opr)

        # 建立前驱-后继关系
        for inp in inputs:
            if hasattr(inp, 'def_') and inp.def_ is not None:
                opr.preds.append(inp.def_)
                inp.def_.succs.append(opr)

        outputs.def_ = opr

        return self.visit(let.body)

    def visit_if(self, _):
        raise NotImplemented

    def visit_global_var(self, gv: relay.GlobalVar):
        if gv.name_hint in self._name2val:
            return self._name2val[gv.name_hint]

        g = Global(self._cvt_type(gv.checked_type), gv.name_hint)
        self._globals.append(g)
        self._name2val[gv.name_hint] = g
        return g.value_

    def visit_op(self, _):
        raise NotImplemented

    def visit_ref_create(self, r: relay.RefCreate):
        inputs = [self.visit(r.value)]
        outputs = Value(self._cvt_type(r.checked_type))
        opr = Operation(OpRegistry.get('ref'), [], inputs, [outputs])
        self._oprs.append(opr)

        # 建立前驱-后继关系
        for inp in inputs:
            if hasattr(inp, 'def_') and inp.def_ is not None:
                opr.preds.append(inp.def_)  # 初始值的定义操作是当前操作的前驱
                inp.def_.succs.append(opr)  # 当前操作是初始值定义操作的后继

        outputs.def_ = opr

        self._outrefs.append(outputs)
        return outputs

    def visit_ref_write(self, r: relay.RefWrite):
        ref = self.visit(r.ref)
        value = self.visit(r.value)
        inputs = [ref, value]
        outputs = Value(self._cvt_type(r.ref.checked_type))
        opr = Operation(OpRegistry.get('ref_write'), [], inputs, [outputs])
        self._oprs.append(opr)

        # 建立前驱-后继关系
        for inp in inputs:
            if hasattr(inp, 'def_') and inp.def_ is not None:
                opr.preds.append(inp.def_)
                inp.def_.succs.append(opr)

        outputs.def_ = opr
   
        self._outrefs = [outputs if outref is ref else outref for outref in self._outrefs]
        return outputs

    def visit_ref_read(self, r: relay.RefRead):
        inputs = [self.visit(r.ref)]
        outputs = Value(self._cvt_type(r.checked_type))
        opr = Operation(OpRegistry.get('ref_read'), [], inputs, [outputs])
        self._oprs.append(opr)

        # 建立前驱-后继关系
        for inp in inputs:
            if hasattr(inp, 'def_') and inp.def_ is not None:
                opr.preds.append(inp.def_)  # 引用的定义操作是当前操作的前驱
                inp.def_.succs.append(opr)  # 当前操作是引用定义操作的后继

        outputs.def_ = opr

        return outputs

    def visit_constructor(self, _):
        raise NotImplemented

    def visit_match(self, _):
        raise NotImplemented

    def _cvt_type(self, ty: relay.Type):
        if isinstance(ty, relay.TensorType):
            return TensorType(self._cvt_ir_value(ty.shape), DataType.from_str(ty.dtype))
        elif isinstance(ty, relay.TypeVar):
            varty = VarTensorType(str(ty.name_hint))
            self._typevars[ty.name_hint] = varty
            return varty
        elif isinstance(ty, relay.FuncType):
            return FuncTensorType([self._cvt_type(at) for at in ty.arg_types], self._cvt_type(ty.ret_type))
        elif isinstance(ty, relay.RefType):
            return RefTensorType(self._cvt_type(ty.value))
        elif isinstance(ty, relay.TupleType):
            return TupleTensorType([self._cvt_type(ft) for ft in ty.fields])
        else:
            raise Exception(f'Cannot handel relay type {str(ty)} ({type(ty)}).')

    def _cvt_ir_value(self, val) -> ValueType:
        if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
            return val.value
        elif isinstance(val, runtime.String):
            return str(val)
        elif isinstance(val, (list, ir.Array)):
            return tuple(self._cvt_ir_value(e) for e in val)
        else:
            return val
    
# def _cvt_type(ty: relay.Type):
#     if isinstance(ty, relay.TensorType):
#         return TensorType(_cvt_ir_value(ty.shape), DataType.from_str(ty.dtype))
#     elif isinstance(ty, relay.TypeVar):
#         return VarTensorType(str(ty.name_hint))
#     elif isinstance(ty, relay.FuncType):
#         return FuncTensorType([_cvt_type(at) for at in ty.arg_types], _cvt_type(ty.ret_type))
#     elif isinstance(ty, relay.RefType):
#         return RefTensorType()
#     elif isinstance(ty, relay.TupleType):
#         return TupleTensorType([_cvt_type(ft) for ft in ty.fields])
#     else:
#         raise Exception(f'Cannot handel relay type {str(ty)} ({type(ty)}).')

# def _cvt_ir_value(val) -> ValueType:
#     if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
#         return val.value
#     elif isinstance(val, runtime.String):
#         return str(val)
#     elif isinstance(val, (list, ir.Array)):
#         return tuple(_cvt_ir_value(e) for e in val)
#     else:
#         return val