import os
import typing
from subprocess import run
from numpy.random import Generator, PCG64
from typing import Iterable, Tuple
import tvm
from tvm import relay, tir, runtime, ir
# from .serenity import simu_mem_serenity
from .serenity_eval import simu_mem_serenity
#from .hmcos_test import simu_mem_hmcos

# def viz2file(filePath:str):
#     '''
#     Visualize the relay graph of the target code file, and save to the same directory.
#     '''
#     cmd = ['python', '/home/nie/RelayOpt/draw_graph.py', f'-f={filePath}']
#     env = os.environ.copy()
#     run(cmd, env=env, check=True, timeout=20, stderr=open(os.devnull, 'w'), stdout=open(os.devnull, 'w'))

def viz2file(filePath:str):
    '''
    Visualize the relay graph of the target code file, and save to the same directory.
    '''
    from tvm import relay
    from GenCoG_cl.gencog.graph.relay import build_graph
    from GenCoG_cl.gencog.graph.viz import visualize

    with open(filePath) as f:
        mod = relay.parse(f.read())
    
    gmod = build_graph(mod)
    visualize(gmod['main'], os.path.basename(filePath)[:-4], os.path.dirname(filePath))

def load_gmod_from_file(filePath:str):
    from tvm import relay
    from GenCoG_cl.gencog.graph.relay import build_graph
    with open(filePath, 'r') as f:
        mod = relay.parse(f.read())
    gmod = build_graph(mod)
    return gmod
        

def compare_code(filePath1:str, filePath2:str, rng: Generator):
    from tvm import relay
    from .graph import GraphAbstractor, GraphComparer

    with open(filePath1, 'r') as f:
        mod1 = relay.parse(f.read())
    with open(filePath2, 'r') as f:
        mod2 = relay.parse(f.read())
    abs1 = GraphAbstractor('graph1').get_graph_from_mod(mod1, rng)
    abs2 = GraphAbstractor('graph2').get_graph_from_mod(mod2, rng)

    comparer = GraphComparer(abs1, abs2)
    return comparer.compare()

def compare_graph(graph1, graph2) -> bool:
    '''
    graph1, graph2: GenCoG.gencog.graph.Graph
    '''
    from .graph import GraphAbstractor, GraphComparer
    abs1 = GraphAbstractor('graph1').abstract(graph1)
    abs2 = GraphAbstractor('graph2').abstract(graph2)
    comparer = GraphComparer(abs1, abs2)
    return comparer.compare()

def exec_and_compare(filePath1: str, filePath2: str, random_seed: int = 43) -> bool:
    '''
    Given the same inputs, compare the exection results of two file.
    '''
    from tvm.relay import parse, create_executor
    from tvm.relay.testing import rand

    with open(filePath1, 'r') as f:
        mod1 = parse(f.read())
    with open(filePath2, 'r') as f:
        mod2 = parse(f.read())
    
    main_fn1 = mod1['main']
    main_fn2 = mod2['main']

    rng = Generator(PCG64(seed=random_seed))
    inputs = gen_tensor_value_dict(main_fn1.params, rng)
    if len(main_fn1.params) != len(main_fn2.params):
        return False
    inputs = list(inputs.values())
    # print(inputs)
    
    output1 = create_executor(mod=mod1).evaluate(main_fn1)(*inputs)
    output2 = create_executor(mod=mod2).evaluate(main_fn2)(*inputs)
    try:
        compare_exec_output(output1, output2)
    except:
        return False
    return True

def compare_exec_output(output1, output2):
    from tvm.testing import assert_allclose
    if isinstance(output1, tvm.runtime.ndarray.NDArray):
        # print(output1, output2)
        assert_allclose(output1.numpy(), output2.numpy())
    elif isinstance(output1, tvm.runtime.container.ADT):
        for o1, o2 in zip(output1, output2):
            compare_exec_output(o1, o2)
    # else:
        # raise Exception(f'Cannot handle output type {type(output1)}.')
        
dtypeMem ={
    'int8':1, 'int16':2, 'int32':4, 'int64':8, 'int1':1, 
    'uint8':1, 'uint16':2, 'uint32':4, 'uint64':8,
    'float16':2, 'float32':4, 'float64':8,
}

# def simu_mem_footprint(g) -> float:
#     '''
#     g: GenCoG_cl.gencog.graph.Graph
#     Simulate the memory footprint of a Graph object.
#     '''
#     from .graph import GraphAbstractor
#     def mul_tuple(tup):
#         res = 1
#         for e in tup:
#             res *= e
#         return res

#     abs = GraphAbstractor('graph').abstract(g)
#     memory = 0
#     for n, ndata in abs.nodes.items():
#         if ndata['type'] == 'tensor':
#             # Check if this value is produced by a let operation. If so, pass it.
#             parents = list(abs.predecessors(n))
#             if len(parents) != 0:
#                 assert len(parents) == 1
#                 pn = parents[0]
#                 assert abs.nodes[pn]['type'] == 'op'
#                 if abs.nodes[pn]['op'] in ['let', 'func', 'ret']:
#                     continue

#             memory += dtypeMem[ndata['dtype']] * mul_tuple(ndata['shape'])
#     return memory / 1024. /1024.

# def simu_mem_from_relay(mod, params) -> float:
#     '''
#     Simulate the memory footprint of a relay mod object and its parameters. 
#     '''
#     from GenCoG_cl.gencog.graph.relay import build_graph
#     graph = build_graph(mod, params)
#     return simu_mem_footprint(graph)

def opt_single_pass(filePath:str, outPath:str, passName:str, rng:Generator, root:str = './'):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(root, 'python')
    cmd = f'python /home/nie/RelayOpt/opt_single_pass.py -i={filePath} -o={outPath} -p={passName} -s={rng.integers(2 ** 63)}'
    try:
        run(cmd, env=env, shell=True, check=True, timeout=60, stderr=open(os.devnull, 'w'), stdout=open(os.devnull, 'w'))
    except:
        # print(f'Run pass "{passName}" failed on file {filePath}.')
        raise Exception(f'Run pass "{passName}" failed on file {filePath}.')
    
def code_valid_check(filePath:str, rng:Generator, opt_level:int = 0, root:str = './'):
    '''
    Check if a relay code file can be compiled, run and compute correctly.
    If valid return True, else return False.
    '''
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(root, 'python')
    cmd = ['python3', './_run_random_ps.py', f'-d={os.path.dirname(filePath)}', f'-o={opt_level}', f'-s={rng.integers(2 ** 63)}']
    try:
        run(cmd, env=env, check=True, timeout=60, stderr=open(os.devnull, 'w'))
    except:
        return False
    return True

def gen_tensor_value(var_ty, rng: Generator):
    if isinstance(var_ty, relay.TensorType):
        return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)
    elif isinstance(var_ty, relay.TupleType):
        return tuple([gen_tensor_value(f, rng) for f in var_ty.fields])
    elif isinstance(var_ty, relay.TypeVar):
        shape = (2, 2)
        dtype = 'float32'
        return rng.uniform(size=[int(d) for d in shape]).astype(dtype)
    else:
        raise Exception(f'Cannot handle relay type {type(var_ty)}.')

def gen_tensor_value_dict(params, rng: Generator):
    return {var.name_hint: gen_tensor_value(var.checked_type, rng) for var in params}

default_seq_dict = {
    0: tvm.ir.transform.Sequential(
        [
            # relay.transform.SimplifyInference(),
        ],
        opt_level=0,
    ), 
    1: tvm.ir.transform.Sequential(
        [
            relay.transform.SimplifyInference(),
            relay.transform.FuseOps(),
        ],
        opt_level=1,
    ), 
    2: tvm.ir.transform.Sequential(
        [
            relay.transform.SimplifyInference(),
            relay.transform.FuseOps(),
            relay.transform.FoldConstant(),
        ],
        opt_level=2,
    ), 
    3: tvm.ir.transform.Sequential(
        [
            relay.transform.SimplifyInference(),
            relay.transform.FuseOps(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            relay.transform.AlterOpLayout(),
            relay.transform.CanonicalizeOps(),
            relay.transform.CanonicalizeCast(),
            relay.transform.EliminateCommonSubexpr(),
        ],
        opt_level=3,
    ), 
    4: tvm.ir.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ToBasicBlockNormalForm(),
            relay.transform.Legalize(),
            relay.transform.SimplifyInference(),
            relay.transform.EliminateCommonSubexpr(),
            relay.transform.CombineParallelConv2D(),
            relay.transform.CombineParallelDense(),
            relay.transform.CombineParallelBatchMatmul(),

            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            relay.transform.SimplifyExpr(),
            relay.transform.CanonicalizeCast(),
            relay.transform.CanonicalizeOps(),
            relay.transform.FlattenAtrousConv(),
            relay.transform.AlterOpLayout(),
            relay.transform.SimplifyExpr(),
            
            relay.transform.FastMath(),
            relay.transform.FoldConstant(),
            relay.transform.SplitArgs(-1),
            relay.transform.FuseOps(),
        ],
        opt_level=4,
    ), 
}

def simu_mem_footprint(g) -> float:
    '''
    Simulate the static memory footprint.
    Note that the memory usage of the following types of nodes is ignored.
        1. The nodes inside an fn (caused by FuseOps).
        2. The virtual nodes, including 'let', 'tuple', 'getitem'.

    Ignore memory footprint inside fn (caused by FuseOps).
    g: GenCoG_cl.gencog.graph.Graph
    Simulate the memory footprint of a Graph object.
    '''
    from .graph.abs import GraphAbsForMem

    abs = GraphAbsForMem('graph').abstract(g)

    # for n, ndata in abs.nodes.items():
    #     print('---')
    #     print(n, ndata)
    #     for p in abs.predecessors(n):
    #         print(p, n, abs[p][n])

    # Label virtual nodes as "no need to alloc memory."
    ignore_nodes = []
    for n, ndata in abs.nodes.items():
        if ndata['type'] != 'op':
            continue

        if ndata['op'] in ('let', 'tuple', 'getitem'):
            for tn in abs.successors(n):
                assert abs.nodes[tn]['type'] == 'tensor'
                if tn not in ignore_nodes:
                    ignore_nodes.append(tn)
            continue

        # if ndata['op'] == 'func':
        #     q = [n]
        #     visited = []
        #     while q:
        #         current = q.pop(0)
        #         visited.append(current)
        #         data = abs.nodes[current]
        #         if data['type'] == 'op' and data['op'] == 'ret':
        #             continue
        #         if data['type'] == 'tensor' and current not in ignore_nodes:
        #             ignore_nodes.append(current)
        #         for child in abs.successors(current):
        #             if child not in visited:
        #                 q.append(child)
                
    memory_bits = 0
    for n, ndata in abs.nodes.items():
        if ndata['type'] == 'tensor' and n not in ignore_nodes:
            # print(ndata['mem'])
            # for parent in abs.predecessors(n):
            #     print(abs.nodes[parent])
            memory_bits += ndata['mem']
    return memory_bits /8. / 1024. /1024.

def simu_mem_from_relay(mod) -> float:
    '''
    Calculate the static memory footprint of a relay mod object. 
    '''
    from GenCoG_cl.gencog.graph.relay import build_graph
    gmod = build_graph(mod)
    return simu_mem_footprint(gmod['main'])

def serenity_mem_from_relay(mod, time_limit = 15) -> float:
    '''
    Calculate the dynamic (serenity) memory footprint of a relay mod object. 
    '''
    from GenCoG_cl.gencog.graph.relay import build_graph
    gmod = build_graph(mod)
    return simu_mem_serenity(gmod['main'], time_limit)

def hmcos_mem_from_relay(mod, time_limit = 15) -> float:
    '''
    Calculate the dynamic (hmcos) memory footprint of a relay mod object. 
    '''
    from GenCoG_cl.gencog.graph.relay import build_graph
    gmod = build_graph(mod)
    return simu_mem_hmcos(gmod['main'], time_limit)

def cal_tvm_mem(mod):
    '''
    Profiling the memory footprint according to TVM memory plan strategy.
    '''
    mod = relay.transform.InferType()(mod)
    func = mod["main"]
    mod = relay.transform.InferType()(mod)
    memory_plan = relay.backend._backend.GraphPlanMemory(func)

    storage_ids = set()
    device_types = set()
    storage_sizes = {}

    # dict = {}

    # body_shape = func.checked_type
    for k, v in memory_plan.expr_to_storage_info.items():
        if tvm.ir.structural_equal(func.body, k):
            continue
        # if k.checked_type == body_shape:
        #     if len(str(k).split('\n')) == len(str(func).split('\n')):
        #         continue
        #     else:
        #         print(k)
        
        # print('---k---')
        # print(k)
        # print(f'Sizes: {_tvm_expr_mem(k.checked_type)}')
        # print('---v---')
        # for x in v.storage_ids:
        #     print(x)
        #     print(v.storage_sizes)
        
        # dict[str(k)] = {}
        for x in v.storage_ids:
            storage_ids.add(x)
            # sizes = sum(v.storage_sizes)
            # sizes = max(v.storage_sizes)
            # sizes = v.storage_sizes[-1]     # TODO: check this.
            sizes = _tvm_expr_mem(k.checked_type)
            if sizes > storage_sizes.get(x, 0):
                storage_sizes[x] = sizes

            # dict[str(k)][str(x)] = str([int(i) for i in v.storage_sizes])
        for x in v.device_types:
            device_types.add(x)
    
    assert len(device_types) == 1
    # print(storage_sizes)

    totol_bytes = 0
    for k, v in storage_sizes.items():
        totol_bytes += v
    
    # import json
    # with open('/home/nie/RelayOpt/out/combine-20230927-225140/9/mem.json', 'w') as f:
    #     json.dump(dict, f, indent='')

    return int(totol_bytes) /1024. /1024.

def _tvm_expr_mem(ty):
    if isinstance(ty, relay.TensorType):
        shape = _cvt_ir_value(ty.shape)
        dtype_bits = _tvm_dtype_bits(ty.dtype)
        return _mul_tuple(shape) * dtype_bits / 8.0
    elif isinstance(ty, relay.TupleType):
        return sum([_tvm_expr_mem(ft) for ft in ty.fields])

def _mul_tuple(tup):
    res = 1
    for e in tup:
        res *= e
    return res

def _tvm_dtype_bits(dtype):
    dtype = str(dtype)
    for code in ('int', 'uint', 'float', 'bfloat'):
        if dtype.find(code) != 0:
            continue
        bits = int(dtype[len(code):])
        return bits
    if dtype == 'bool':
        return 1
    raise ValueError(
        f'Cannot create DataType from \'{dtype}\''
    )

def _cvt_ir_value(val):
    if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return val.value
    elif isinstance(val, runtime.String):
        return str(val)
    elif isinstance(val, (list, ir.Array)):
        return tuple(_cvt_ir_value(e) for e in val)
    else:
        return val

# TODO: Simulate tvm dynamic memory allocation -- reversed postorder traversal