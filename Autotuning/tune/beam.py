'''
Use beam search to direct the optimization process.
'''

import os
from numpy.random import Generator
from typing import Dict, List, Tuple
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
    
class Beam:
    def __init__(self, codePath: str, beamSize: int, rng: Generator, profiler = simu_mem_from_relay) -> None:
        self._rng = rng
        self.code_path_ = codePath
        self.beam_size_ = beamSize
        self.profiler_ = profiler

        # The path where the template files are stored.
        self.workspace_ = os.path.join(os.path.dirname(codePath), 'beam')
        if os.path.exists(self.workspace_):
            os.system(f'rm -rf {self.workspace_}')
        os.mkdir(self.workspace_)

        # Init the work space.
        status = self._get_status(self.code_path_)
        assert isinstance(status[-1], float)
        os.system(f'cp {self.code_path_} {os.path.join(self.workspace_, "0.txt")}')
        self.code_num = 1

        self._start_mem = status[-1]

        # Init code_lu and status_lu
        self.status_lu = [status]
        self.code_lu = {os.path.join(self.workspace_, "0.txt"): status}
        self.code_seq = {os.path.join(self.workspace_, "0.txt"): []}

    def run(self, epochs: int, verbose: bool = False):
        self._current = [os.path.join(self.workspace_, "0.txt")]
        self._best_mem = self._start_mem
        self._seq = []
        for epoch in tqdm(range(epochs)):
            if verbose:
                print(f'######## Epoch {epoch} ########')
                for code_path in self._current:
                    print(os.path.basename(code_path), self.code_lu[code_path], self.code_seq[code_path])
            if not self._one_iter():
                break
        
        print(f'Beam: Best mem is {self._best_mem} achieved by {self._seq}')
        return self._best_mem, self._seq
    
    def _one_iter(self) -> bool:
        # Choose the best #n optimization choice.
        candicates = [] # Optimization candidates, containing (code_path, pass_name, optimized_mem).
        for code_path in self._current:
            for pass_name in RelayPassTable.NameTable:
                tmp_path = os.path.join(os.path.dirname(self.code_path_), 'tmp.txt')
                mem = _opt_pass_simu_mem(code_path, tmp_path, pass_name, self._rng.integers(2 ** 63), self.profiler_)
                if mem is not None:
                    candicates.append((code_path, pass_name, mem))
        
        if len(candicates) == 0:
            return False
        candicates = sorted(candicates, key=lambda x:x[-1])
        candicates = candicates[:self.beam_size_]
        print(f'Candicates num {len(candicates)}')
        
        # Optimize with the best #n candidates.
        new_current = []
        best_mem = float('inf')
        best_code = None
        for pick_code, pick_pass, after_mem in candicates:
            new_code = os.path.join(self.workspace_, f'{self.code_num}.txt')
            mem = _opt_pass_simu_mem(pick_code, new_code, pick_pass, self._rng.integers(2 ** 63), self.profiler_)
            status = self._get_status(new_code)
            assert mem == after_mem # and mem == status[-1], f'{mem}, {after_mem}, {status[-1]}'
            self.code_num += 1

            # Check if this candidate is Done. 
            # 1. It is already explored.
            # 2. It cannot improve the memory.
            if status in self.status_lu or mem >= self.code_lu[pick_code][-1]:
                continue

            self.status_lu.append(status)
            self.code_lu[new_code] = status
            self.code_seq[new_code] = self.code_seq[pick_code] + [pick_pass]
            
            new_current.append(new_code)
            if mem < best_mem:
                best_mem = mem
                best_code = new_code

        if len(new_current) == 0:
            return False

        assert best_mem < self._best_mem
        self._best_mem = best_mem
        self._seq = self.code_seq[best_code]
        self._current = new_current
        
        return True
    
    def _get_status(self, codePath: str) -> Tuple[float]:
        with open(codePath, 'r') as f:
            mod = parse(f.read())
        graph = load_gmod_from_file(codePath)['main']
        status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), self.profiler_(mod))
        return status