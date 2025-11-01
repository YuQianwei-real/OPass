import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError, check_output
from sys import stdout
from time import strftime

from numpy.random import Generator, PCG64
from tqdm import tqdm


args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-d', '--directory', type=str, default='/home/nie/RelayOpt/out/ReBench/', help='Codes directory.')
    p.add_argument('-s', '--seed', type=int, default=55, help='Random seed of graph generator.')
    p.add_argument('-e', '--epochs', type=int, default=200, help='Total iteration number.')
    args = p.parse_args()

def main():
    rng = Generator(PCG64(seed=args.seed))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')

    failed = []
    success = []
    dirnames = os.listdir(args.directory)
    dirnames = sorted(dirnames)
    done = True
    for dn in dirnames:
        # if dn == '5':
        #     done = False
        # if done:
        #     continue

        print('#################')
        print(f'Tuning case {dn}...')
        case_path = os.path.join(args.directory, dn)
        if os.path.exists(os.path.join(case_path, 'tune_results_serenity.json')):
            print('Found serenity tune results')
            continue
      # if os.path.exists(os.path.join(case_path, 'tune_results.json')):
       #    print('Found tune results.')
       #    continue

        code_path = os.path.join(case_path, 'code.txt')
        cmd = ['python3', 'run_transfer_graph.py', f'-p={code_path}', f'-e={args.epochs}', f'-s={rng.integers(2 ** 63)}']
        try:
            run(cmd, env=env, check=True, timeout=1800, stderr=open(os.devnull, 'w'))
            success.append(' '.join(cmd))
        except:
            print(' '.join(cmd))
            failed.append(' '.join(cmd))
    
    print('failed', failed)
    print('success', success)

if __name__ == '__main__':
    parse_args()
    main()
