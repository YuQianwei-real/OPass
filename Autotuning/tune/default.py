'''
Use TVM default seq to direct the optimization process.
'''

import os
import math
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

default_seq_4 = [
    'RemoveUnusedFunctions',
    'ToBasicBlockNormalForm',
    'Legalize',
    'SimplifyInference',
    'EliminateCommonSubexpr',
    'CombineParallelConv2D',
    'CombineParallelDense',
    'CombineParallelBatchMatmul',
    'FoldConstant',
    'FoldScaleAxis',
    'SimplifyExpr',
    'CanonicalizeCast',
    'CanonicalizeOps',
    'FlattenAtrousConv',
    'AlterOpLayout',
    'SimplifyExpr',
    'FastMath',
    'FoldConstant',
    'SplitArgs',
    'FuseOps',
]

class DefaultTVM:
    def __init__(self, codePath: str, rng: Generator, profiler = simu_mem_from_relay) -> None:
        # ------- Some parameters -------
        # ------- Some parameters -------

        self._rng = rng
        self.code_path_ = codePath
        self.profiler_ = profiler

        # The path where the template files are stored.
        self.workspace_ = os.path.join(os.path.dirname(codePath), 'default')
        if os.path.exists(self.workspace_):
            os.system(f'rm -rf {self.workspace_}')
        os.mkdir(self.workspace_)

        # Init the work space.
        with open(self.code_path_, 'r') as f:
            mod = parse(f.read())
        graph = load_gmod_from_file(self.code_path_)['main']
        status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), self.profiler_(mod))
        assert isinstance(status[-1], float)
        self._start_code = os.path.join(self.workspace_, "0.txt")
        self._start_mem = status[-1]
        os.system(f'cp {self.code_path_} {self._start_code}')
        self.code_num = 1

    def run(self):
        self._current_code = self._start_code
        self._current_mem = self._start_mem
        for pn in default_seq_4:
            new_code = os.path.join(self.workspace_, f'{self.code_num}.txt')
            mem = _opt_pass_simu_mem(self._current_code, new_code, pn, self._rng.integers(2 ** 63), self.profiler_)
            self.code_num += 1

            if mem is None:
                continue
            self._current_code = new_code
            self._current_mem = mem
        print(f'Default TVM: {self._current_mem} mb.')

        return self._current_mem, default_seq_4