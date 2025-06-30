import sys
from argparse import Namespace, ArgumentParser
from numpy.random import Generator, PCG64

import tvm
from tvm import relay

from Autotuning.sequence import RelaySeq, RelayPassSelector
from Autotuning.util import simu_mem_from_relay, cal_tvm_mem, serenity_mem_from_relay, hmcos_mem_from_relay

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-s', '--seed', type=int, default=58, help='Random seed of graph generator.')
    p.add_argument('-i', '--input', type=str, default='./ReBench/2/code.txt', help='Input file path.')
    p.add_argument('-o', '--output', type=str, default='', help='Output file path.')
    p.add_argument('-p', '--passname', type=str, default='FuseOps', help='Pass name.')
    p.add_argument('-m', '--profiler', type=str, default='hmcos', help='Memory profiler name.')
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
    
    
    passSelector = RelayPassSelector(rng)
    
    try:
        p = passSelector.wrap_pass(args.passname)
    except Exception as e:
        exit(1)
    
    relaySeq = RelaySeq()
    relaySeq.append(passSelector.wrap_pass('InferType'))
    relaySeq.append(p)
    
    try:
        with tvm.transform.PassContext(opt_level=5):
            mod = relaySeq.seq(mod)
    except:
        exit(1)
    
    try:
        static_mod = relay.transform.DynamicToStatic()(mod)
        mem = profiler(static_mod)      
        if not isinstance(mem, float):
            exit(1)
    except:
        exit(1)
    #print(5)
    if args.output != '':
        with open(args.output, 'w') as f:
            f.write(mod.astext())

        try:
            with open(args.output, 'r') as f:
                mod = relay.parse(f.read())
        except:
            exit(1)

    print(f'{mem} mb')

if __name__ == '__main__':
    parse_args()
    main()
