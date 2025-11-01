from numpy.random import Generator, PCG64
from typing import List
from argparse import Namespace, ArgumentParser
from tvm.relay import parse, transform, Var, build
import tvm
import json
import os
import time

from Autotuning.tune.transfer import TranferGraph
# from Autotuning.tune.transfer_eval import TranferGraph
from Autotuning.tune.greedy import Greedy
from Autotuning.tune.beam import Beam
from Autotuning.tune.sa import SimulatedAnnealing
from Autotuning.tune.default import DefaultTVM

from Autotuning.util import cal_tvm_mem, simu_mem_from_relay, serenity_mem_from_relay, hmcos_mem_from_relay

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-p', '--path', type=str, default='/home/yqw/OPass/ReBench/36/code.txt', help='Code path.')
    p.add_argument('-s', '--seed', type=int, default=55, help='Random seed of graph generator.')
    p.add_argument('-e', '--epochs', type=int, default=200, help='Total iteration number.')
    p.add_argument('-m', '--profiler', type=str, default='static', help='Memory profiler name.')
    args = p.parse_args()

def gen_tensor_value(var: Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)

def gen_tensor_value_dict(params: List[Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}

def main():
    rng = Generator(PCG64(seed=args.seed))

    with open(args.path, 'r') as f:
        mod = parse(f.read())
    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)
    with open(args.path+'.txt', 'w') as f:
        f.write(mod.astext())

    if args.profiler == 'static':
        profiler = simu_mem_from_relay
        #profiler = simu_mem_from_relay
    elif args.profiler == 'serenity':
        profiler = serenity_mem_from_relay
    elif args.profiler == 'hmcos':
        profiler = hmcos_mem_from_relay
    else:
        exit(1)

    # print(profiler(mod), 'MB')
    # print(simu_mem_from_relay(mod), 'MB')
    # print(cal_tvm_mem(mod), 'MB')

    # compilation check
    # main_fn = mod['main']
    # params = gen_tensor_value_dict(main_fn.params[1:], rng)
    # with tvm.transform.PassContext(opt_level=4) as PC:
    #     _ = build(mod, target='llvm', params=params)


    tune_results = {}
    tune_results['Origin'] = (profiler(mod), 'None')
    print('Origin:', tune_results['Origin'][0], 'mb')

    time_list = {}

    s_time = time.time()
    default = DefaultTVM(args.path+'.txt', rng, profiler=profiler)
    mem, seq = default.run()
    d_time = time.time()
    tune_results['Default'] = (mem, str(seq), d_time -s_time)

    time_list['default'] = d_time -s_time

    s_time = time.time()
    greedy = Greedy(args.path+'.txt', rng, profiler=profiler)
    mem, seq = greedy.run(args.epochs)
    d_time = time.time()
    tune_results['Greedy'] = (mem, str(seq), d_time -s_time)

    time_list['greedy'] = d_time -s_time
    
    s_time = time.time()
    beam = Beam(args.path+'.txt', 3, rng, profiler=profiler)
    mem, seq = beam.run(args.epochs, True)
    d_time = time.time()
    tune_results['Beam'] = (mem, str(seq), d_time -s_time)

    time_list['beam'] = d_time -s_time

    s_time = time.time()
    sa = SimulatedAnnealing(args.path+'.txt', rng, profiler=profiler)
    mem, seq = sa.run(int(args.epochs/20), 20)
    d_time = time.time()
    tune_results['SA'] = (mem, str(seq), d_time -s_time)

    time_list['sa'] = d_time -s_time

    s_time = time.time()
    transferG = TranferGraph(args.path+'.txt', rng, profiler=profiler) # , profiler=cal_tvm_mem
    mem, seq = transferG.run(args.epochs)
    d_time = time.time()
    tune_results['Transfer'] = (mem, str(seq), d_time -s_time)

    time_list['transfer'] = d_time -s_time
    print(time_list)
    # d_time = time.time()
    # cost = d_time - s_time
    # print('Time:', cost, 's') 
    # method: static, dynamic
    # greedy: 105.628   1314.441
    # beam search: 252.751   2511.418
    # transfer: 494.865    4801.917
    # SA: 167.992    2255.854

    if profiler == serenity_mem_from_relay:
        res_fn = 'tune_results_serenity.json'
    elif profiler == hmcos_mem_from_relay:
        res_fn = 'tune_results_hmcos.json'
    else:
        res_fn = 'tune_results.json'
    with open(os.path.join(os.path.dirname(args.path), res_fn), 'w') as f:
        json.dump(tune_results, f, indent='')

    # TODO(Done): check why tvm mem > static mem -- because of dumplicated memory allocation of let

if __name__ == '__main__':
    parse_args()
    try:
        main()
    except:
        exit(1)
