import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError, check_output
import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm
import json
import psutil

args = Namespace()
pass_dir = '/home/yqw/RelayOpt/out/test/single_pass/'

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-d', '--directory', type=str, default='/home/nie/RelayOpt/out/combine-20230608-032017/', help='Case directory.')
    p.add_argument('-s', '--seed', type=int, default=52, help='Random seed of graph generator.')
    args = p.parse_args()

def get_current_memory_gb() -> float:
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. # / 1024.

def resolve_script_res(r):
    r = str(r, 'utf-8').split('\n')
    r_mem = r[0].strip()
    r_time = r[1].strip()
    assert r_mem.endswith(' mb')
    assert r_time.endswith(' ms')
    exec_mem = float(r_mem[:-3].split('/')[0])
    simu_mem = float(r_mem[:-3].split('/')[1])
    exec_time = float(r_time[:-3])
    return exec_mem, simu_mem, exec_time

def main():
    rng = Generator(PCG64(seed=args.seed))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')

    for case_dir in os.listdir(args.directory):
        case_path = os.path.join(args.directory, case_dir)
        print('##########################')
        print('Testing', case_path)
        
        # Record the test result and mark this case as tested
        if os.path.exists(os.path.join(case_path, 'TESTED')):
            # os.system('rm -rf ' + os.path.join(case_path, 'TESTED'))
            continue
        with open(os.path.join(case_path, 'TESTED'), 'w') as _:
            pass
        tes_res = {}
        try:
            cmd = ['python3', './_run_random_ps.py', f'-d={case_path}', f'-o=0', f'-s={rng.integers(2 ** 63)}', '-M', '-T']
            r = check_output(cmd, env=env, timeout=1200, stderr=open(os.devnull, 'w'))
            exec_mem_ref, simu_mem_ref, exec_time_ref = resolve_script_res(r)
            print(f'Unoptimized: {exec_mem_ref}/{simu_mem_ref} mb; {exec_time_ref} ms.')
            tes_res['Unoptimized'] = {'mem': exec_mem_ref, 'sim_mem':simu_mem_ref, 'time': exec_time_ref}

            cmd = ['python3', './_run_random_ps.py', f'-d={case_path}', f'-o=4', f'-s={rng.integers(2 ** 63)}', '-M', '-T']
            r = check_output(cmd, env=env, timeout=1200, stderr=open(os.devnull, 'w'))
            exec_mem, simu_mem, exec_time = resolve_script_res(r)
            print(f'DefaultOpt: {exec_mem}/{simu_mem} mb; {exec_time} ms.')
            tes_res['DefaultOpt'] = {'mem': exec_mem, 'sim_mem':simu_mem, 'time': exec_time}
            
            for fn in os.listdir(pass_dir):
                seq_path = os.path.join(pass_dir, fn)
                try:
                    cmd = ['python3', './_run_random_ps.py', f'-d={case_path}', f'-q={seq_path}', f'-s={rng.integers(2 ** 63)}', '-M', '-T']
                    r = check_output(cmd, env=env, timeout=1200, stderr=open(os.devnull, 'w'))
                    exec_mem, simu_mem, exec_time = resolve_script_res(r)
                    # abs(exec_mem - exec_mem_ref) / exec_mem_ref > 0.05 or \
                    if abs(exec_time - exec_time_ref) / exec_time_ref > 0.05 or \
                        abs(simu_mem - simu_mem_ref) / simu_mem_ref > 0.05:
                        print(f'{fn}: {exec_mem}/{simu_mem} mb; {exec_time} ms.')
                    tes_res[fn] = {'mem': exec_mem, 'sim_mem':simu_mem, 'time': exec_time}
                except:
                    # print('Error:', ' '.join(cmd))
                    print('Failed', fn)
                    tes_res[fn] = {}
                    pass
            
            with open(os.path.join(case_path, 'mem_test.json'), 'w') as f:
                json.dump(tes_res, f, indent='')
        except:
            print('Error:', ' '.join(cmd))
            continue

if __name__ == '__main__':
    parse_args()
    main()

