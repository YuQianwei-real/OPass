from time import time
from networkx import DiGraph
from typing import List, cast, Tuple, Optional

from .graph.abs import GraphAbsForMem

def simu_mem_serenity(g, time_limit: Optional[int] = None) -> float:
    '''
    g: GenCoG_cl.gencog.graph.Graph
    time_limit: seconds -- if the simutation costs more than time_limit, then raise an Error.
    Simulate the memory footprint of a Graph object by Serenity's algorithm (MLSys'20).
    '''
    G:DiGraph = GraphAbsForMem('graph').abstract(g)
    G, N = _preprocess(G)
    
    # initialize memoization
    s = []
    mem_cur, mem_peak = 0, 0
    z = _zero_indegree(s, G)
    M = {z:(s, mem_cur, mem_peak)}

    start_t = time()
    # iterate each step
    for _ in range(N):
        end_t = time()
        if time_limit is not None and end_t - start_t > time_limit:
            # raise Exception('Time out when simulate memory footprint by serenity.')
            # raise TimeoutError('Time out when simulate memory footprint by serenity.')
            ret = 0
            for k in M.keys():
                if ret < M[k][2]:
                    ret = M[k][2]
            return ret

        M_next = {}
        for z in M:
            s, mem_cur, mem_peak = M[z]

            # If 'cast' exists in s, firstly execute 'cast'. This can reduce search time and do not affect the schedule result.
            exist_cast = False
            for u in z:
                if G.nodes[u]['op'] == 'cast':
                    exist_cast = True
                    s_next = s.copy()
                    s_next.append(u)
                    z_next = _zero_indegree(s_next, G)
                    mem_next = mem_cur + _alloc_memory(u, G)
                    mem_peak_next = max(mem_next, mem_peak)
                    
                    # deallocate the memory which will never be used
                    zero_outs = _zero_outdegree(s_next, G)
                    for p in _predecessors(u, G):
                        if p in zero_outs:
                            mem_next -= _alloc_memory(p, G)

                    # memoize schedule with least peak memory
                    if z_next not in M_next or (z_next in M_next and mem_peak_next < M_next[z_next][2]):
                        M_next[z_next] = (s_next, mem_next, mem_peak_next)
                    break
            if exist_cast:
                continue

            for u in z:
                s_next = s.copy()
                s_next.append(u)
                z_next = _zero_indegree(s_next, G)
                mem_next = mem_cur + _alloc_memory(u, G)
                mem_peak_next = max(mem_next, mem_peak)
                
                # deallocate the memory which will never be used
                zero_outs = _zero_outdegree(s_next, G)
                for p in _predecessors(u, G):
                    if p in zero_outs:
                        mem_next -= _alloc_memory(p, G)

                # memoize schedule with least peak memory
                if z_next not in M_next or (z_next in M_next and mem_peak_next < M_next[z_next][2]):
                    M_next[z_next] = (s_next, mem_next, mem_peak_next)

                #  print(s_next, mem_next, mem_peak_next, z_next, zero_outs)
        M = M_next

    return M[()][2]


def _preprocess(G:DiGraph) -> DiGraph:
    '''
    Preprocess the graph:
    1. Add input op node before all input tensor node, as well as constants and globals. 
    2. Delete virtual nodes, including 'let'.
    '''
    opr_num = 0
    input_tensors = []
    let_oprs = []
    for n in G:
        if cast(str, n).startswith('in') or cast(str, n).startswith('glob') or cast(str, n).startswith('const'):
            assert G.nodes[n]['type'] == 'tensor'
            assert len(list(G.predecessors(n))) == 0
            input_tensors.append(n)

            opr_num += 1

        elif cast(str, n).startswith('opr'):
            assert G.nodes[n]['type'] == 'op'

            if G.nodes[n]['op'] == 'let':
                let_oprs.append(n)
                continue

            opr_num += 1

        elif cast(str, n).startswith('imm') or cast(str, n).startswith('out'):
            assert G.nodes[n]['type'] == 'tensor'
            assert len(list(G.predecessors(n))) == 1
    
    for n in input_tensors:
        G.add_node('opr'+n, type='op', op='input', attrs={})
        G.add_edge('opr'+n, n, order=0, direction='out')

    G = _del_lets(G, let_oprs)

    return G, opr_num

def _del_lets(G:DiGraph, let_oprs:List[str]):
    for n in let_oprs:
        ins = list(G.predecessors(n))
        assert len(ins) == 1
        in_value = ins[0]

        in_ops = list(G.predecessors(in_value))
        assert len(in_ops) == 1
        in_op = in_ops[0]

        outs = list(G.successors(n))
        assert len(outs) == 1
        out_value = outs[0]

        assert G.nodes[in_value]['mem'] == G.nodes[out_value]['mem']

        G.add_edge(in_op, out_value, order=G[in_op][in_value]['order'], direction='out')
        G.remove_node(in_value)
        G.remove_node(n)
    return G

def _zero_indegree(s:List[str], G:DiGraph) -> Tuple[str]:
    '''
    Find the op nodes in 'G'\'s' with zero indegree based on already scheduled nodes 's'.
    '''
    # Find tensors generated by already scheduled nodes
    gened_tensors = []
    for n in s:
        for c in G.successors(n):
            assert G.nodes[c]['type'] == 'tensor'
            gened_tensors.append(c)

    # Find those nodes whose in-tensors all have been generated.
    zero_in_nodes = []
    for n, ndata in G.nodes.items():
        if ndata['type'] != 'op' or n in s:
            continue

        indegree = 0
        for p in G.predecessors(n):
            assert G.nodes[p]['type'] == 'tensor'
            if p not in gened_tensors:
                indegree += 1

        if indegree == 0:
            zero_in_nodes.append(n)
    return tuple(sorted(zero_in_nodes))

def _zero_outdegree(s:List[str], G:DiGraph) -> List[str]:
    '''
    Find the op nodes in 's' with zero outdegree based on already scheduled nodes 's'.
    '''
    zero_out_nodes = []
    for n, ndata in G.nodes.items():
        if ndata['type'] != 'op' or n not in s:
            continue

        outdegree = 0
        for c in G.successors(n):
            assert G.nodes[c]['type'] == 'tensor'
            for cc in G.successors(c):
                assert G.nodes[cc]['type'] == 'op'
                if cc not in s:
                    outdegree += 1
        
        if outdegree == 0:
            zero_out_nodes.append(n)
    return zero_out_nodes


def _alloc_memory(u:str, G:DiGraph) -> float:
    '''
    Calculate the memory footprint of op node 'u', i.e., the out-tensors of u.
    '''
    assert G.nodes[u]['type'] == 'op'

    if G.nodes[u]['op'] in ('let', 'tuple', 'getitem'):
        return 0.

    memory_bits = 0
    for n in G.successors(u):
        assert G.nodes[n]['type'] == 'tensor'

        # size = 1
        # for dim in G.nodes[n]['shape']:
        #     size *= dim
        # memory += size * dtypeMem[G.nodes[n]['dtype']]
        memory_bits += G.nodes[n]['mem']
    return memory_bits /8. /1024. /1024.

def _predecessors(u:str, G:DiGraph) -> List[str]:
    '''
    Find the 'op' predecessors of u in G. Assume these preds are all in s.
    '''
    assert G.nodes[u]['type'] == 'op'

    preds = []
    for p in G.predecessors(u):
        assert G.nodes[p]['type'] == 'tensor'
        for pp in G.predecessors(p):
            if pp not in preds:
                preds.append(pp)
    return preds