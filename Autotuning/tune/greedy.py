'''
Use greedy method to direct the optimization process.
'''

import os
from numpy.random import Generator
from typing import Dict, List
from tqdm import tqdm
from subprocess import check_output
from tvm.relay import parse
from Autotuning.util import load_gmod_from_file, simu_mem_from_relay, cal_tvm_mem, serenity_mem_from_relay, hmcos_mem_from_relay
from Autotuning.sequence import RelayPassTable

def _opt_pass_simu_mem(codePath: str, outPath: str, passName: str, seed: int, profiler):
    env = os.environ.copy()
    #env['PYTHONPATH'] = os.path.join('./', 'python')

    if profiler == simu_mem_from_relay:
        profiler_name = 'static'
    elif profiler == hmcos_mem_from_relay:
        profiler_name = 'hmcos'
    elif profiler == serenity_mem_from_relay:
        profiler_name = 'serenity'
    else:
        raise Exception(f'No such profiler {profiler}')

    try:
        cmd = ['python3', './_opt_pass_simu_mem.py', f'-i={codePath}', f'-o={outPath}', f'-p={passName}', f'-s={seed}', f'-m={profiler_name}']
        r = check_output(cmd, env=env, timeout=60, stderr=open(os.devnull, 'w'))
        r = str(r, 'utf-8').strip()
        assert r.endswith(' mb')
        random_mem = float(r[:-3])
        return random_mem
    except:
        # print(' '.join(cmd))
        return None
    
class Greedy:
    def __init__(self, codePath: str, rng: Generator, profiler = simu_mem_from_relay) -> None:
        self._rng = rng
        self.code_path_ = codePath
        self.profiler_ = profiler

        # The path where the template files are stored.
        self.workspace_ = os.path.join(os.path.dirname(codePath), 'greedy')
        if os.path.exists(self.workspace_):
            os.system(f'rm -rf {self.workspace_}')
        os.mkdir(self.workspace_)

        # Init the work space.
        with open(self.code_path_, 'r') as f:
            mod = parse(f.read())
        start_mem = self.profiler_(mod)
        assert isinstance(start_mem, float)
        os.system(f'cp {self.code_path_} {os.path.join(self.workspace_, "0.txt")}')
        self.code_num = 1

        self._start_mem = start_mem

    def run(self, epochs: int):
        self._current = os.path.join(self.workspace_, "0.txt")
        self._best_mem = self._start_mem
        self._seq = []
        #print('start')
        
        for _ in tqdm(range(epochs)):
            new_code = self._one_iter()
            if new_code == self._current:
                break
            else:
                self._current = new_code
        
        print(f'Greedy: Best mem is {self._best_mem} achieved by {self._seq}')
        return self._best_mem, self._seq
    
    def _one_iter(self) -> str:
        # Choose the best optimization pass.
        best_mem = float('inf')
        pick_pass = None
        for pass_name in RelayPassTable.NameTable:
            tmp_path = os.path.join(os.path.dirname(self.code_path_), 'tmp.txt')
            mem = _opt_pass_simu_mem(self._current, tmp_path, pass_name, self._rng.integers(2 ** 63), self.profiler_)
            #print(pass_name, mem)
            if mem is not None and mem < best_mem:
                best_mem = mem
                pick_pass = pass_name
        if pick_pass is None or best_mem >= self._best_mem:
            return self._current
        
        # Optimize with the picked pass.
        new_code = os.path.join(self.workspace_, f'{self.code_num}.txt')
        #print('new_code:',new_code)
        mem = _opt_pass_simu_mem(self._current, new_code, pick_pass, self._rng.integers(2 ** 63), self.profiler_)
        self._best_mem = mem
        self.code_num += 1
        self._seq.append(pick_pass)
        print(mem, pick_pass, new_code)
        return new_code