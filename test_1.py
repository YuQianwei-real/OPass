import sys, os
import subprocess
from argparse import Namespace, ArgumentParser
from numpy.random import Generator, PCG64
from networkx import DiGraph
from typing import List, cast, Tuple, Optional, Dict

import tvm
from tvm import relay

from GenCoG_cl.gencog.graph.base import Operation, Vertex, Input, Output,Constant
from GenCoG_cl.gencog.graph.viz import visualize
from GenCoG_cl.gencog.graph.relay import build_graph, print_relay, build_func_graph, GraphBuilder
from GenCoG_cl.gencog.hmcos.mem import estimate_peak_hmcos, estimate_peak_hmcos0, estimate_peak_networkx
from GenCoG_cl.gencog.hmcos.hier import HierGraph, Sequence, HierInput, Group
from GenCoG_cl.gencog.hmcos.join import JoinSequencePass, MakeGroupPass
from GenCoG_cl.gencog.hmcos.sched import SerenitySchedule, HierarchicalSchedule

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

def process_all_folders_with_output():
    """处理ReBench下所有36个文件夹并实时显示输出"""
    
    # 遍历1到36的所有文件夹
    for i in range(18, 37):
        folder_path = f"./ReBench/{i}"
        file_path = f"{folder_path}/code.txt"
        result_path = f"{folder_path}/tune_results_serenity.json"
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            #if os.path.exists(result_path):
            #    print('serenity result exits')
            #    continue
            
            print(f"\n{'='*50}")
            print(f"正在处理文件夹 {i}: {file_path}")
            print(f"{'='*50}")
            
            try:
                # 使用Popen实时显示输出
                process = subprocess.Popen([
                    sys.executable,
                    "run_transfer_graph.py",
                    "-p", file_path,
                    "-m", "serenity"
                ])
                
                # 等待进程完成
                return_code = process.wait()
                
                if return_code == 0:
                    print(f"\n✓ 文件夹 {i} 处理完成\n")
                else:
                    print(f"\n✗ 文件夹 {i} 处理失败，返回码: {return_code}\n")
                    
            except Exception as e:
                print(f"✗ 文件夹 {i} 处理时发生异常: {e}")
                
        else:
            print(f"⚠ 警告: 文件 {file_path} 不存在，跳过文件夹 {i}")   

if __name__ == '__main__':
    #parse_args()
    #main()
    process_all_folders_with_output()

    networkx_total, hmcos0_total, hmcos_total = 0, 0, 0
    
    with open('./ReBench/6/code.txt', 'r') as f:
        mod = relay.parse(f.read())

    
    static_mod = relay.transform.DynamicToStatic()(relay.transform.InferType()(mod))
    print(cal_tvm_mem(static_mod))
    print(simu_mem_from_relay(static_mod))
    """
    graph = build_graph(static_mod)['main']
    
    opr_num = len(graph.oprs_)
    print('opr_num = ',opr_num)
    op_list = []
    for op in graph.oprs_:
        op_list.append(op.op_.name_)
    print(op_list)
    hier = HierGraph(graph)
    
    sched = SerenitySchedule(graph, joinOps=True, trySimple=False, nSamples=100)
    #sched = HierarchicalSchedule(graph)
    op_list = []
    for op in sched:
        op_list.append(op.op_.name_)
    print(len(op_list), op_list)
    print([op.op_.name_ for op in graph.oprs_ if op not in sched])
    print(estimate_peak_hmcos0(sched, graph))  
  

    """
    
    
    
    


    