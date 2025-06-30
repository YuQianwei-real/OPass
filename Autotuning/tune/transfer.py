'''
Use transfer graph to direct the optimization process.
'''

import os
import math
import networkx as nx
import numpy as np
from numpy.random import Generator
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict as dd
from subprocess import check_output
import json

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
    
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)

class InherentOppoStat:
    def __init__(self, filePath: str) -> None:
        self.should_ihrt_cnt = 0
        self.miss_ihrt_cnt = 0
        self.new_appr_cnt = 0
        self.actual_cnt = 0

        self.cause_miss_ps: Dict[str, Dict[str, int]] = dd(lambda :dd(lambda :0))
        self.gen_new_ps: Dict[str, Dict[str, int]] = dd(lambda :dd(lambda :0))

        self.file_path_ = filePath

    def stat(self, ihrt_oppos: List[str], actual_oppos: List[str], pass_name: str):
        self.should_ihrt_cnt += len(ihrt_oppos)

        missed = False
        for pn in ihrt_oppos:
            if pn not in actual_oppos:
                self.miss_ihrt_cnt += 1
                self.cause_miss_ps[pass_name][pn] += 1
                missed = True
        if missed:
            self.cause_miss_ps[pass_name]['cnt'] += 1

        self.actual_cnt += len(actual_oppos)
        new_appear = False
        for pn in actual_oppos:
            if pn not in ihrt_oppos:
                self.new_appr_cnt += 1
                self.gen_new_ps[pass_name][pn] += 1
                new_appear = True
        if new_appear:
            self.gen_new_ps[pass_name]['cnt'] += 1

    def save(self):
        obj = {
            'should_ihrt_cnt': self.should_ihrt_cnt,
            'miss_ihrt_cnt': self.miss_ihrt_cnt,
            'new_appr_cnt': self.new_appr_cnt,
            'actual_cnt': self.actual_cnt,

            'cause_miss_ps': self.cause_miss_ps,
            'gen_new_ps': self.gen_new_ps,
        }

        with open(self.file_path_, 'w') as f:
            json.dump(obj, f, indent='')

    def show(self):
        print(f'Inherent rate: {(self.should_ihrt_cnt - self.miss_ihrt_cnt) / self.should_ihrt_cnt}.')
        print(f'Newly generation rate: {self.new_appr_cnt / self.actual_cnt}')

PassPrior = dd(lambda :1)
PassPrior.update({
    'EliminateCommonSubexpr': 4,
    'SimplifyExpr': 4, 
    'PartialEvaluate': 4,
    'DeadCodeElimination': 4,
    'FuseOps': 3,
    'ToMixedPrecision': 3,
})

class TranferGraph:
    def __init__(self, codePath: str, rng: Generator, profiler = simu_mem_from_relay) -> None:
        # ------- Some parameters -------
        self._check_p = 0.03    # Possibility of opportunity checking.
        # ------- Some parameters -------

        self._rng = rng
        self.code_path_ = codePath
        self.profiler_ = profiler

        # The path where the template files are stored.
        self.workspace_ = os.path.join(os.path.dirname(codePath), 'transfer_graph')
        if os.path.exists(self.workspace_):
            os.system(f'rm -rf {self.workspace_}')
        os.mkdir(self.workspace_)
        self.tmp_path_ = os.path.join(os.path.dirname(self.code_path_), 'tmp.txt')

        # Init the work space and transfer graph with target code.
        with open(self.code_path_, 'r') as f:
            mod = parse(f.read())
        graph = load_gmod_from_file(self.code_path_)['main']
        status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), self.profiler_(mod))
        assert isinstance(status[-1], float)
        start_code = os.path.join(self.workspace_, "0.txt")
        self.start_code_ = start_code
        self.start_mem_ = status[-1]
        os.system(f'cp {self.code_path_} {start_code}')

        # The code look-up table. Key: string of code's status. Value: List of code path.
        self.code_lu: Dict[str, List[str]] = {str(status): [start_code]}
        self.code_num: str = 1

        # The transfer graph
        # Node attributes:
        #   name: str(status)
        #   stat: status
        #   mem: status[-1]
        #   chosen: number of times selected
        #   score: affect the possibility to be chosen
        #   and more... (e.g. a possibility distribution of optimization pass)
        # Edge attribute:
        #   trans: number of transformation along this edge
        self.TG = nx.DiGraph()
        self.add_node(status)
        # Detect optimization opportunities for the start node.
        self.TG.nodes[str(status)]['oppo'] = list(self._detect_oppo(start_code, str(status)).keys())

        # Record the transformation path. Key: code path. Value: list of pass name to reach this code.
        self.code_seq: Dict[str, List[str]] = {start_code: []}

        # ------- For Experiment -------
        self.IS = InherentOppoStat(os.path.join(os.path.dirname(self.code_path_), 'IS.json'))
        # ------- For Experiment -------

    def run(self, epochs: int, verbose: bool = False):
        self._best_mem = float('inf')
        self._best_mem_epoch = 0
        self._best_mem_code = os.path.join(self.workspace_, "0.txt")
        for epoch in tqdm(range(1, epochs + 1)):
            self._one_iter(epoch, verbose)
        print(f'Transfer: Best mem: {self._best_mem} in {self._best_mem_epoch} epochs by {self.code_seq[self._best_mem_code]}.')
        return self._best_mem, self.code_seq[self._best_mem_code]

    def _one_iter(self, epoch: int, verbose: bool = True):
        # Choose a computation graph for next optimization
        pick_node = self._node_select()
        pick_code = self._code_select(pick_node)
        self.TG.nodes[pick_node]['chosen'] += 1

        # Choose an optimization pass
        pick_pass = self._pass_select(pick_node)
        if pick_pass is None:
            return
        
        # Optimize the chosen code by the chosen pass
        mem = _opt_pass_simu_mem(pick_code, self.tmp_path_, pick_pass, self._rng.integers(2 ** 63), self.profiler_)
        # print(pick_node, pick_code, pick_pass, mem)
        if mem is None:
            modified = self._back_propagate_miss(pick_node, pick_pass)
            self._forward_propagate_miss(modified, pick_pass)
            return

        # Update the workspace
        new_code = os.path.join(self.workspace_, f"{self.code_num}.txt")
        os.system(f'mv {self.tmp_path_} {new_code}')
        self.code_num += 1

        # Update the transfer graph by optimization result, as well as the code_lu and code_seq
        graph = load_gmod_from_file(new_code)['main']
        status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), mem)
        assert isinstance(status[-1], float)
        
        # Update nodes
        new_node = str(status)
        if new_node not in self.TG:
            print(f'Found new node {new_node} from {pick_node} by {pick_pass} in {new_code}.')
            self.add_node(status)
            self.code_lu[new_node] = []
        self.code_lu[new_node].append(new_code)
        self.code_seq[new_code] = self.code_seq[pick_code] + [pick_pass]

        # Update edges
        if (pick_node, new_node) not in self.TG.edges:
            self.add_edge(pick_node, new_node)
        if pick_pass not in self.TG[pick_node][new_node]['passes']:
            self.TG[pick_node][new_node]['passes'].append(pick_pass)

        # Update the oppo of picked node, i.e., delete pick_pass if it is not an opportunity.
        # TODO: propogate to other node
        if new_node == pick_node:
            modified = self._back_propagate_miss(pick_node, pick_pass)
            self._forward_propagate_miss(modified, pick_pass)

        # Update the score of picked node.
        self._update_score(pick_node)

        # Propagate optimization opportunities to new node.
        inherent_oppo: List[str] = self.TG.nodes[pick_node]['oppo']
        current_oppo: List[str] = self.TG.nodes[new_node]['oppo']
        current_oppo += [pn for pn in inherent_oppo if pn not in current_oppo and pn != pick_pass]
        self.TG.nodes[new_node]['oppo'] = current_oppo
        self._update_score(new_node)

        # Randomly check whether some new opportunities appear.
        if self._rng.random() < self._check_p:
            check_oppo = list(self._detect_oppo(new_code, new_node).keys())
            # self.TG.nodes[new_node]['oppo'] = check_oppo
            # print('-----Before-----')
            # for pn in current_oppo:
            #     print(pn)
            # print('-----After-----')
            # for pn in check_oppo:
            #     print(pn)
            # print(f'-----From {pick_pass}-----')
            # self.IS.stat(current_oppo, check_oppo, pick_pass)
            # self.IS.show()
            # self.IS.save()
            self._back_propagate(new_node, check_oppo, current_oppo)
        
        # Record the best case.
        if mem < self._best_mem:
            self._best_mem = mem
            self._best_mem_epoch = epoch
            self._best_mem_code = new_code

        # Print the template results.
        if verbose:
            print(f'Epoch {epoch}: choose {self.TG.nodes[pick_node]["idx"]}, transfer to {self.TG.nodes[new_node]["idx"]}')
            self._print_graph()
    
    def _print_graph(self):
        print('##########')
        for n, ndata in sorted(self.TG.nodes.items(), key=lambda x:x[1]['idx']):
            print(f'Node {ndata["idx"]} {n}: {ndata["mem"]} mb; {ndata["chosen"]} times chosen; \
{len(list(self.TG.successors(n)))} times transfer; score {ndata["score"]}')
            print(ndata['oppo'], ndata['chosen_pass'])
        print('##########')

    def add_node(self, status: Tuple[float]):
        idx = self.TG.number_of_nodes()
        self.TG.add_node(str(status), idx=idx, stat=status, mem=status[-1], 
                         chosen=0, score=1, chosen_pass = [], oppo = [])
    
    def add_edge(self, u, v):
        self.TG.add_edge(u, v, passes=[])

    def _node_select(self) -> str:
        p_dict = {}
        for n, ndata in self.TG.nodes.items():
            # TODO: Calculate potential of each node.
            p_dict[n] = ndata['score']
        p_dict = sorted(p_dict.items(), key=lambda e:e[1], reverse=True)    #order by descending
        rand = self._rng.random(dtype='float')
        length = len(p_dict)
        try:
            index = int(math.floor(math.log(math.pow((1-rand),length),0.05)))
        except:
            index = length -1
        if (index>=length -1):
            index = length - 1
        return p_dict[index][0]

    def _code_select(self, node: str) -> str:
        return self._rng.choice(self.code_lu[node])

    def _pass_select(self, node: str) -> str:
        '''
        Select an optimization pass for a node.
        '''

        # Strategy 0: Randomly selection.
        # pick_pass = self._rng.choice(RelayPassTable.NameTable)

        # Strategy 1: Do not select the passes which have been chosen.
        # candidates = [pn for pn in RelayPassTable.NameTable if pn not in self.TG.nodes[node]['chosen_pass']]
        # if len(candidates) == 0:
        #     return None
        # pick_pass = self._rng.choice(candidates)
        # self.TG.nodes[node]['chosen_pass'].append(pick_pass)

        # Strategy 2: Select pass from available opportunities.
        candidates = [pn for pn in self.TG.nodes[node]['oppo'] if pn not in self.TG.nodes[node]['chosen_pass']]
        if len(candidates) == 0:
            return None
        pick_pass = self._rng.choice(candidates)
        self.TG.nodes[node]['chosen_pass'].append(pick_pass)

        # TODO: S3: add human prior for S2
        # candidates = [pn for pn in self.TG.nodes[node]['oppo'] if pn not in self.TG.nodes[node]['chosen_pass']]
        # if len(candidates) == 0:
        #     return None
        # scores = [PassPrior[pn] for pn in candidates]
        # scores = softmax(scores)
        # pick_pass = self._rng.choice(candidates, p=scores)
        # self.TG.nodes[node]['chosen_pass'].append(pick_pass)
        return pick_pass
        
    def _update_score(self, node: str):
        '''
        Update the score for a node
        '''

        # S1: score = e^(-mem)*(trans/chosen)
        # trans = len(list(self.TG.successors(node)))
        # chosen = self.TG.nodes[node]['chosen']
        # mem = self.TG.nodes[node]['mem']
        # score = math.pow(math.e, - mem) * ((trans + 1) / (chosen + 1))
        # self.TG.nodes[node]['score'] = score

        # Another choice:
        #       better  unchanged   worse
        # new   inc     inc         keep
        # old   keep    dec         dec
        # ...

        # S2: score  = (start_mem / mem) * |opportunities - chosen_oppos|
        mem = self.TG.nodes[node]['mem']
        num_oppo = len(self.TG.nodes[node]['oppo']) - len(self.TG.nodes[node]['chosen_pass'])
        score = (self.start_mem_ / mem) * num_oppo
        self.TG.nodes[node]['score'] = score
        
    def _detect_oppo(self, codePath: str, node: str) -> Dict[str, Tuple[float]]:
        '''
        Detect the optimization opportunity.
        '''
        opportunities = {}
        for pn in RelayPassTable.NameTable:
            mem = _opt_pass_simu_mem(codePath, self.tmp_path_, pn, self._rng.integers(2 ** 63), self.profiler_)
            if mem is None:
                continue
            graph = load_gmod_from_file(self.tmp_path_)['main']
            status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), mem)
            if str(status) != node:
                opportunities[pn] = status
        return opportunities
    
    def _back_propagate(self, node: str, check_oppo: List[str], current_oppo: List[str]):
        '''
        Back propagate newly found opportunities.
        TODO: back propagate disappeared opportunities.
        '''
        new_found_oppo = [pn for pn in check_oppo if pn not in current_oppo]
        missed_oppo = [pn for pn in current_oppo if pn not in check_oppo]

        if len(new_found_oppo) + len(missed_oppo) != 0:
            print('------Start Propagation------')

        for pn in new_found_oppo:
            modifed_nodes = self._back_propagate_oppo(node, pn)
            print(f'Back propogate pass {pn} to {len(modifed_nodes)} nodes.')
            self._forward_propagate_oppo(modifed_nodes, pn)

        for pn in missed_oppo:
            modifed_nodes = self._back_propagate_miss(node, pn)
            print(f'Back propogate missed pass {pn} to {len(modifed_nodes)} nodes.')
            self._forward_propagate_miss(modifed_nodes, pn)

        if len(new_found_oppo) + len(missed_oppo) != 0:
            print('------End Propagation------')

    def _back_propagate_oppo(self, node: str, oppo: str) -> List[str]:
        if oppo in self.TG.nodes[node]['oppo']:
            return
        self.add_oppo(node, oppo)

        to_prop = self.predecessors(node)
        ed_prop = [node]
        modifed = [node]
        while to_prop:
            n = to_prop.pop(0)
            if n in ed_prop:
                continue
            ed_prop.append(n)

            # Check if such opportunity is already existed.
            if oppo in self.TG.nodes[n]['oppo']:
                continue

            # Check if this node have such opportunity.
            code = self._code_select(n)
            mem = _opt_pass_simu_mem(code, self.tmp_path_, oppo, self._rng.integers(2 ** 63), self.profiler_)
            if mem is None:
                continue

            graph = load_gmod_from_file(self.tmp_path_)['main']
            status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), mem)
            if str(status) == n:
                continue

            # If this node have such opportunity and have not been explored.
            self.add_oppo(n, oppo)
            for parent in self.predecessors(n):
                if parent not in to_prop and parent not in ed_prop:
                    to_prop.append(parent)
            modifed.append(n)

        return modifed
    
    def _forward_propagate_oppo(self, nodes: List[str], oppo: str):
        ed_prop = nodes
        for node in nodes:
            to_prop = []
            for child in self.successors(node):
                if child not in to_prop and child not in ed_prop \
                    and oppo not in self.TG[node][child]['passes']:
                    to_prop.append(child)

            while to_prop:
                n = to_prop.pop(0)
                if n in ed_prop:
                    continue
                ed_prop.append(n)
                
                # Check if such opportunity is already existed.
                if oppo in self.TG.nodes[n]['oppo']:
                    continue

                self.add_oppo(n, oppo)
                for child in self.successors(n):
                    if child not in to_prop and child not in ed_prop \
                        and oppo not in self.TG[n][child]['passes']:    # Check if this node is generated by such oppo.
                        to_prop.append(child)

    def _back_propagate_miss(self, node: str, oppo: str) -> List[str]:
        if oppo not in self.TG.nodes[node]['oppo']:
            return
        self.del_oppo(node, oppo)

        to_prop = self.predecessors(node)
        ed_prop = [node]
        modifed = [node]
        while to_prop:
            n = to_prop.pop(0)
            if n in ed_prop:
                continue
            ed_prop.append(n)

            # Check if such opportunity is already removed.
            if oppo not in self.TG.nodes[n]['oppo']:
                continue

            # Check if such opportunity is already explored and the result is saved.
            if oppo in self.TG.nodes[n]['chosen_pass']:
                continue

            # Check if this node have such opportunity. If has, continue, else remove such oppo.
            code = self._code_select(n)
            mem = _opt_pass_simu_mem(code, self.tmp_path_, oppo, self._rng.integers(2 ** 63), self.profiler_)
            if mem is not None:
                graph = load_gmod_from_file(self.tmp_path_)['main']
                status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), mem)
                if str(status) != n:
                    continue

            # Remove the oppo
            self.del_oppo(n, oppo)
            for parent in self.predecessors(n):
                if parent not in to_prop and parent not in ed_prop:
                    to_prop.append(parent)
            modifed.append(n)

        return modifed
    
    def _forward_propagate_miss(self, nodes: List[str], oppo: str):
        ed_prop = nodes
        for node in nodes:
            to_prop = []
            for child in self.successors(node):
                if child not in to_prop and child not in ed_prop:
                    to_prop.append(child)

                    
            while to_prop:
                n = to_prop.pop(0)
                if n in ed_prop:
                    continue
                ed_prop.append(n)
                
                # Check if such opportunity is already removed.
                if oppo not in self.TG.nodes[n]['oppo']:
                    continue

                # Check if such opportunity is already explored and the result is saved.
                if oppo in self.TG.nodes[n]['chosen_pass']:
                    continue

                self.del_oppo(n, oppo)
                for child in self.successors(n):
                    if child not in to_prop and child not in ed_prop:
                        to_prop.append(child)


    def add_oppo(self, node: str, oppo: str):
        self.TG.nodes[node]['oppo'].append(oppo)

    def del_oppo(self, node: str, oppo: str):
        self.TG.nodes[node]['oppo'].remove(oppo)
        if oppo in self.TG.nodes[node]['chosen_pass']:
            self.TG.nodes[node]['chosen_pass'].remove(oppo)

    def successors(self, node: str) -> List[str]:
        return list(self.TG.successors(node))
    
    def predecessors(self, node: str) -> List[str]:
        return list(self.TG.predecessors(node))

'''
            if len(check_oppo) != len(current_oppo):
                print('-----Node-----')
                print(new_node, 'from', pick_node, 'by', pick_pass)
                print('-----Before-----')
                for pn in current_oppo:
                    print(pn)
                print('-----After-----')
                for pn in check_oppo:
                    print(pn)

'''