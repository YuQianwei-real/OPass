import sys
import itertools
from argparse import Namespace, ArgumentParser
from numpy.random import Generator, PCG64
from networkx import DiGraph
from typing import List, cast, Tuple, Optional, Dict

import tvm
from tvm import relay

from GenCoG_cl.gencog.graph.base import Operation, Vertex
from GenCoG_cl.gencog.graph.viz import visualize
from GenCoG_cl.gencog.graph.relay import build_graph, print_relay, build_func_graph, GraphBuilder
from GenCoG_cl.gencog.hmcos.mem import estimate_peak_hmcos, estimate_peak_hmcos0, estimate_peak_networkx

from Autotuning.sequence import RelaySeq, RelayPassSelector
from Autotuning.util import simu_mem_from_relay, cal_tvm_mem, serenity_mem_from_relay, hmcos_mem_from_relay, simu_mem_footprint
from Autotuning.graph.abs import GraphAbsForMem
from Autotuning.serenity_eval import _preprocess, _zero_indegree, _alloc_memory
from Autotuning.hmcos_util import compute_op_lifetimes, estimate_peak,is_valid_sequence


args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-s', '--seed', type=int, default=58, help='Random seed of graph generator.')
    p.add_argument('-i', '--input', type=str, default='./ReBench/3/code.txt', help='Input file path.')
    p.add_argument('-o', '--output',type=str, default='./ReBench/3/output', help='Output file path.')
    p.add_argument('-p', '--passname', type=str, default='FuseOps', help='Pass name.')
    p.add_argument('-m', '--profiler', type=str, default='serenity', help='Memory profiler name.')
    args = p.parse_args()


def main():
    rng = Generator(PCG64(seed=args.seed))

    if args.profiler == 'static':
        profiler = simu_mem_from_relay
    elif args.profiler == 'tvm':
        profiler = cal_tvm_mem
    elif args.profiler == 'serenity':
        profiler = serenity_mem_from_relay
    elif args.profiler == 'hmcos':
        profiler = hmcos_mem_from_relay
    else:
        exit(1) 

    with open(args.input, 'r') as f:
        mod = relay.parse(f.read())

    
    static_mod = relay.transform.DynamicToStatic()(relay.transform.InferType()(mod))
    #print('static_mod:', mod)
    graph_mod = build_graph(static_mod)['main']
    #visualize(graph_mod, 'static_mod', args.output)

    op_seq = graph_mod.oprs_
    
    print(f'hmcos_mem = {estimate_peak_hmcos(op_seq, graph_mod)} mb')
    op_name = []
    for op in op_seq:
        in_memo = 0
        in_use_cnt = []
        op_name.append(op.op_.name_)
        if op.op_.name_ in ('let', 'tuple', 'getitem'):
            for in_ in op.inputs_:
                in_memo += in_.type_.memo_bytes
                in_use_cnt.append(len(in_.uses_))
            print(op.op_.name_, in_memo/1024/1024, 
                  op.outputs_[0].type_.memo_bytes /1024 /1024, in_use_cnt)
    
    print(op_name)
    
    
    passSelector = RelayPassSelector(rng)
    #'''
    try:
        p = passSelector.wrap_pass(args.passname)
    except Exception as e:
        exit(1)
    
    relaySeq = RelaySeq()
    relaySeq.append(passSelector.wrap_pass('InferType'))
    relaySeq.append(p)
    
    try:
        with tvm.transform.PassContext(opt_level=5):
            fuseop_mod = relaySeq.seq(mod) 
    except:
        exit(1)

    try:
        infer_mod = relay.transform.InferType()(mod)
        
        static_mod = relay.transform.DynamicToStatic()(infer_mod)
        #visualize(static_mod, 'static_mod', './ReBench/2/output')
        g_graph = build_graph(static_mod)['main']
        
        mem = profiler(static_mod)
        print(f'serenity_mem {mem} mb')
        G = GraphAbsForMem('graph').abstract(g_graph)
        op_seqence = []
        for n, ndata in G.nodes.items():
            if ndata['type'] == 'op':
                op_seqence.append(n)
        mem = estimate_peak_networkx(G, op_seqence)
        print(f'networkx_mem = {mem} mb')

        
    except:
        exit(1)
    
    """
    if args.output != '':
        with open(args.output+'/test_result.txt', 'w') as f:
            f.write(mod.astext())

        try:
            with open(args.output+'/test_result.txt', 'r') as f:
                mod = relay.parse(f.read())
        except:
            exit(1)
    """
    

if __name__ == '__main__':
    #parse_args()
    #main()
    networkx_total, hmcos0_total, hmcos_total = 0, 0, 0
    

    with open('./test/resnet18_train.txt', 'r') as f:
        mod = relay.parse(f.read())
    static_mod = relay.transform.DynamicToStatic()(relay.transform.InferType()(mod))
    graph = build_graph(static_mod)['main']

    abs = GraphAbsForMem('graph').abstract(graph)
    op_seq = []
    for n in abs.nodes:
        if abs.nodes[n]['type'] == 'op':
            op_seq.append(n)
    
    #networkx_mem = estimate_peak_networkx(op_seq, abs)
    #hmcos0_mem = estimate_peak_hmcos0(graph.oprs_, graph)
    hmcos_mem = estimate_peak_hmcos(graph.oprs_, graph)
    #print('networkx_mem = ', networkx_mem )
    #print('hmcos0_mem = ', hmcos0_mem)
    print('hmcos_mem = ', hmcos_mem)
        