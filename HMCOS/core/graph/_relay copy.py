from typing import Dict, List, cast, Iterable

import numpy as np
from tvm import relay, tir, runtime, ir
from tvm.ir import IRModule

from .base import GraphVisitor, Value, Graph, VertexKind, Input, Output, Operation
from ..solve.solver import RefType, TensorTypeVar, TensorType, FuncType
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


def print_relay(g: Graph):
    return RelayPrinter().print(g)


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
        return fmt_val([])
    else:
        assert False


class RelayPrinter(GraphVisitor[None]):
    def __init__(self):
        super().__init__()
        self._buf = CodeBuffer()
        self._val_names: Dict[Value, str] = {}
        self._arg_gen = NameGenerator('%x')
        self._res_gen = NameGenerator('%')

    def print(self, g: Graph):
        # Function signature
        self._buf.writeln('#[version = "0.0.5"]')
        self._buf.write('def @main')
        self._buf.write_pos(
            map(lambda i: lambda: self._buf.write(
                f'{self.visit_value(i.value_)}: {i.value_.type_}'), g.inputs_)
        )
        self._buf.write(' -> ')
        self._buf.write_pos(
            map(lambda o: lambda: self._buf.write(str(o.value_.type_)), g.outputs_)
        )

        # Function body
        self._buf.writeln(' {')
        with self._buf.indent():
            for out in g.outputs_:
                self.visit(out)
            out_str = str(tuple(self.visit_value(o.value_) for o in g.outputs_)).replace('\'', '')
            self._buf.writeln(out_str)
        self._buf.writeln('}')

        return str(self._buf)

    def visit_input(self, i: Input):
        return

    def visit_output(self, o: Output):
        return self.visit(o.value_.def_)

    def visit_operation(self, opr: Operation):
        # Visit predecessors
        for v in opr.inputs_:
            self.visit(v.def_)

        # Print output value
        op_name = opr.op_.name_
        tup_out = len(opr.outputs_) > 1 or op_name in tuple_out_ops
        if tup_out:
            tup_name = self._res_gen.generate()
            self._buf.write(tup_name)
        else:
            tup_name = ''
            self._buf.write(self.visit_value(opr.outputs_[0]))
        self._buf.write(' = ')

        # Print operator call
        self._buf.write(opr.op_.name_)
        args = map(lambda v: self.visit_value(v), opr.inputs_)
        if op_name in tuple_in_ops:
            arg_str = str(tuple(args)).replace('\'', '')
        else:
            arg_str = ', '.join(args)
        self._buf.write_pos([
            lambda: self._buf.write(arg_str),
            lambda: self._buf.write_named(
                map(lambda a: (a[0], lambda: self._buf.write(fmt_val(a[1]))), opr.attrs_),
                prefix='', suffix=''
            )
        ])
        ty_str = repr(tuple(out.type_ for out in opr.outputs_)) if tup_out else repr(
            opr.outputs_[0].type_)
        self._buf.writeln(f'; /* ty={ty_str} */')

        # Unpack tuple
        if tup_out:
            for i, v in enumerate(opr.outputs_):
                self._buf.writeln(f'{self.visit_value(v)} = {tup_name}.{i};')

    def visit_value(self, v: Value):
        if v in self._val_names:
            return self._val_names[v]
        if v.def_.kind == VertexKind.IN:
            name = self._arg_gen.generate()
        else:
            name = self._res_gen.generate()
        self._val_names[v] = name
        return name


def build_graph(mod: IRModule, params: Dict[str, np.ndarray]):
    return GraphBuilder(params).build(mod['main'])


class GraphBuilder(relay.ExprFunctor):
    def __init__(self, params: Dict[str, np.ndarray]):
        super().__init__()
        self._params = params
        self._name2val: Dict[str, Value] = {}
        self._inputs: List[Input] = []
        self._oprs: List[Operation] = []

        self._name2func: Dict[str, relay.function.Function] = {}

    def build(self, fn: relay.Function):
        # Create inputs
        self._inputs = [Input(_cvt_type(var.checked_type), var.name_hint in self._params)
                        for var in fn.params]
        self._name2val = {p.name_hint: inp.value_ for p, inp in zip(fn.params, self._inputs)}

        # Build operations
        if isinstance(fn.body, (relay.Call, relay.TupleGetItem, relay.Var)):
            outputs = [Output(self.visit(fn.body))]
        elif isinstance(fn.body, relay.Tuple):
            visit_res = [self.visit(f) for f in fn.body.fields]
            outputs = [Output(v) for v in _flatten(visit_res)]
        elif isinstance(fn.body, relay.expr.Let):
            outputs = [Output(f) for f in self.visit(fn.body)]
        else:            
            raise TypeError('{} not supported.'.format(type(fn.body).__name__))

        # Create graph
        return Graph(self._inputs, outputs, self._oprs)

    def visit_function(self, fn: relay.Function):
        # Create inputs
        # self._inputs = [Input(_cvt_type(var.checked_type), var.name_hint in self._params)
        #                 for var in fn.params]
        # self._name2val.update({p.name_hint: inp.value_ for p, inp in zip(fn.params, self._inputs)})

        # Build operations
        # return self.visit(fn.body)

        self._func_name2val: Dict[str, Value] = {}
        self._name2val_buffer = self._name2val
        self._name2val = self._func_name2val

        # Create funcDef node along with params generated by funcDef
        param_values = [Value(_cvt_type(var.checked_type)) for var in fn.params]
        self._name2val.update({p.name_hint: inp for p, inp in zip(fn.params, param_values)})
        opr = Operation(OpRegistry.get('func'), [], [], _flatten(param_values))
        self._oprs.append(opr)

        # Visit function body   TODO: Check
        rets = self.visit(fn.body)

        # Create funcRet node
        funcValue = Value(FuncType())
        opr = Operation(OpRegistry.get('ret'), [], _flatten(rets), [funcValue])
        self._oprs.append(opr)

        self._name2val = self._name2val_buffer
        return funcValue

    def visit_var(self, var: relay.Var):
        return self._name2val[var.name_hint]

    def visit_constant(self, const: relay.Constant):
        inp = Input(_cvt_type(const.checked_type), True)
        self._inputs.append(inp)
        return inp.value_

    def visit_tuple(self, tup: relay.Tuple):
        return [self.visit(f) for f in tup.fields]

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        item = self.visit(getitem.tuple_value)
        return item[getitem.index]

        
    def visit_call(self, call: relay.Call):
        if isinstance(call.op, relay.function.Function):
            name = 'func'
            inputs = [self.visit(a) for a in call.args]
            inputs = _flatten(inputs)
            outputs = [get_outputs_from_type(var.checked_type) for var in call.op.params]

            # Convert attribute values
            if call.op.attrs is None or (not hasattr(call.op.attrs, 'keys')):
                attrs = []
            else:
                attrs = [(str(k), _cvt_ir_value(call.op.attrs[k])) for k in call.op.attrs.keys()]
            
            opr = Operation(OpRegistry.get(name), attrs, inputs, _flatten(outputs))
            self._oprs.append(opr)

            for o, p in zip(outputs, call.op.params):
                self._name2val[p.name_hint] = o
            func_outs = self.visit(call.op.body)

            # Create return node
            name = 'ret'
            inputs = _flatten(func_outs)
            outputs = get_outputs_from_type(call.checked_type)
            opr = Operation(OpRegistry.get(name), [], inputs, _flatten(outputs))
            self._oprs.append(opr)
            return outputs
            
        if isinstance(call.op, relay.Var):      # this Var is a function's name
            fn = call.op.name_hint
            assert fn in self._name2func
            func = self._name2func[fn]

            # Create call node
            name = 'func'
            inputs = [self.visit(a) for a in call.args]
            inputs = _flatten(inputs)
            outputs = _flatten([get_outputs_from_type(var.checked_type) for var in func.params])
            outputs = [get_outputs_from_type(var.checked_type) for var in func.params]
            # outputs = [Value(_cvt_type(var.checked_type)) for var in func.params]
            if func.attrs is None or (not hasattr(func.attrs, 'keys')):
                attrs = []
            else:
                attrs = [(str(k), _cvt_ir_value(func.attrs[k])) for k in func.attrs.keys()]
            opr = Operation(OpRegistry.get(name), attrs, inputs, _flatten(outputs))
            self._oprs.append(opr)

            # Create body node
            for o, p in zip(outputs, func.params):
                self._name2val[p.name_hint] = o
            func_outs = self.visit(func.body)

            # Create return node
            name = 'ret'
            inputs = _flatten(func_outs)
            outputs = get_outputs_from_type(call.checked_type)
            opr = Operation(OpRegistry.get(name), [], inputs, _flatten(outputs))
            self._oprs.append(opr)
            return outputs
        
        if isinstance(call.op, relay.RefRead):
            fn = self.visit(call.op)
            inputs = [fn] + [self.visit(a) for a in call.args]
            out_ty = call.checked_type
            outputs = get_outputs_from_type(out_ty)

            # Create operation
            opr = Operation(OpRegistry.get('call'), [], _flatten(inputs), _flatten(outputs))
            self._oprs.append(opr)
            return outputs
        
        # Collect input values
        name = call.op.name
        inputs = [self.visit(a) for a in call.args]
        inputs = _flatten(inputs)

        # Convert attribute values
        if call.attrs is None or (not hasattr(call.attrs, 'keys')):
            attrs = []
        else:
            attrs = [(str(k), _cvt_ir_value(call.attrs[k])) for k in call.attrs.keys()]

        # Create output values
        out_ty = call.checked_type
        outputs = get_outputs_from_type(out_ty)

        # Create operation
        opr = Operation(OpRegistry.get(name), attrs, inputs, _flatten(outputs))
        self._oprs.append(opr)

        return outputs

    def visit_let(self, let: relay.expr.Let):
        # Collect input values
        # print('---------')
        # print(let)
        # print('####')
        # print(type(let.value))
        # print(let.value)
        # print('####')
        # print(type(let.var))
        # print(let.var)
        # print('####')
        # print(type(let.body))
        # print(let.body)

        if isinstance(let.value, relay.Function):
            self._name2func[let.var.name_hint] = let.value
            return _flatten(self.visit(let.body))
        
        inputs = self.visit(let.value)
        inputs = _flatten(inputs)

        # Create output values
        out_ty = let.var.checked_type
        outputs = get_outputs_from_type(out_ty)
        self._name2val[let.var.name_hint] = outputs
        
        # Create operation
        opr = Operation(OpRegistry.get('let'), [], inputs, _flatten(outputs))
        self._oprs.append(opr)

        return _flatten(self.visit(let.body))

    def visit_if(self, _):
        raise NotImplemented

    def visit_global_var(self, _):
        raise NotImplemented

    def visit_op(self, _):
        raise NotImplemented

    def visit_ref_create(self, r: relay.RefCreate):
        inputs = self.visit(r.value)
        outputs = Value(RefType())
        opr = Operation(OpRegistry.get('ref_create'), [], _flatten(inputs), _flatten(outputs))
        self._oprs.append(opr)
        return outputs

    def visit_ref_write(self, r: relay.RefWrite):
        ref = self.visit(r.ref)
        value = self.visit(r.value)
        inputs = [ref, value]
        opr = Operation(OpRegistry.get('ref_write'), [], _flatten(inputs), [])
        self._oprs.append(opr)
        return ref

    def visit_ref_read(self, r: relay.RefRead):
        inputs = [self.visit(r.ref)]
        outputs = get_outputs_from_type(r.checked_type)
        opr = Operation(OpRegistry.get('ref_read'), [], inputs, _flatten(outputs))
        self._oprs.append(opr)
        return outputs

    def visit_constructor(self, _):
        raise NotImplemented

    def visit_match(self, _):
        raise NotImplemented

def _cvt_type(ty: relay.TensorType):
    if isinstance(ty, relay.TypeVar):
        return TensorTypeVar(str(ty.name_hint))
    return TensorType(_cvt_ir_value(ty.shape), DataType.from_str(ty.dtype))

def _cvt_ir_value(val) -> ValueType:
    if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return val.value
    elif isinstance(val, runtime.String):
        return str(val)
    elif isinstance(val, (list, ir.Array)):
        return tuple(_cvt_ir_value(e) for e in val)
    else:
        return val
    
def _flatten(data):  # 定义递归函数  
    if not isinstance(data, Iterable):
        return [data]
    
    sum_data = []  
    for i in data:  
        if isinstance(i, Iterable):  # 如果i是可迭代的对象（列表等），调用函数本身；直到执行else语句  
            sum_data.extend(_flatten(i))  
        else:  
            sum_data.append(i)  
      
    return sum_data  

def get_outputs_from_type(out_ty):
    if isinstance(out_ty, relay.TensorType):
        outputs = Value(_cvt_type(out_ty))
    elif isinstance(out_ty, ir.type_relation.TypeCall):
        # outputs = [Value(_cvt_type(f)) for f in out_ty.args]
        print(dir(out_ty))
        raise
    elif isinstance(out_ty, relay.TupleType):
        outputs = [get_outputs_from_type(f) for f in out_ty.fields]
    # elif isinstance(out_ty, ir.type.FuncType):
    #     outputs = get_outputs_from_type(out_ty.ret_type)
    elif isinstance(out_ty, relay.RefType):
        outputs = Value(RefType())
    elif isinstance(out_ty, relay.FuncType):
        outputs = Value(FuncType())
    else:
        raise TypeError('{} not supported.'.format(type(out_ty)))   # .__name__
    return outputs