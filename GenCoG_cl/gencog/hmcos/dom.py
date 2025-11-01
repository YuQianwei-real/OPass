from __future__ import annotations
import sys
from typing import Any, Dict, List

from ..graph.base import Vertex, Graph, Operation, Value, GraphVisitor
from .hier import HierVertex, HierGraph
from .util import AddUnique




class DomNode:
    """Node in the dominator tree."""
    def __init__(self, v: 'HierVertex'):
        self.hierVertex_ = v  # Reference to the original vertex
        self.parent: DomNode = None    # Parent in dominator tree
        self.children: List[DomNode] = []    # Children in dominator tree
        self.in_num = 0       # In index for dominance check
        self.out_num = 0      # Out index for dominance check

    def Dominates(self, other: DomNode, strict=False):
        """Check if this node dominates another."""
        if strict:
            return self.in_num < other.in_num and self.out_num > other.out_num
        else:
            return self.in_num <= other.in_num and self.out_num >= other.out_num


class DfNode:
    """Node in the depth-first spanning tree (auxiliary structure)."""
    NONE = -1 # Constant to represent "no node" or "undefined"

    def __init__(self, v: HierVertex):
        self.hierVertex_: HierVertex = v
        self.parent: int = self.NONE   # DFS树父节点索引
        self.semi:int = self.NONE     # 半支配编号
        self.bucket: List[int] = []      # List of nodes whose semi-dominator is this node半支配桶
        self.idom: int = self.NONE    # 即时支配者索引
        self.ancestor: int = self.NONE # 并查集祖先
        self.best: int = 0             # 最优路径节点
        self.size: int = 0             # 子树大小
        self.child: int = self.NONE    # 孩子索引


class DomBuilder:
    """Builds the dominator tree using the Lengauer-Tarjan algorithm."""
    def __init__(self, hier: HierGraph):
        self.hier_graph: HierGraph = hier
        self.nodes: List[DfNode] = []           # DfNode数组
        self.vertIdx = {}         # 顶点到索引的映射

    
    def Build(self, root):
        """构建支配树"""
        assert root is not None, "root is None"
        
        # 1. 深度优先遍历收集所有节点
        self.nodes.clear()
        #self.vertIdx.clear()
        count = 0
        
        for hier_V in self.hier_graph.DfsHierRange():
            df_node = DfNode(hier_V)
            df_node.parent = DfNode.NONE
            df_node.semi = DfNode.NONE
            df_node.idom = DfNode.NONE
            df_node.ancestor = DfNode.NONE
            df_node.best = count
            df_node.size = 0
            self.nodes.append(df_node)
            self.vertIdx[hier_V] = count
            count += 1
            #print(count)


        if len(self.nodes) <= 1:
            print("Graph is trivial. No need to build dominator tree.")
            return []

        # 2. 初始化semi和父节点关系
        for v in range(len(self.nodes)):
            self.nodes[v].semi = v
            # 设置DFS树的父节点
            for w_vert in self.nodes[v].hierVertex_.succs:
                if w_vert in self.vertIdx:
                    w = self.vertIdx[w_vert]
                    if self.nodes[w].semi == DfNode.NONE:
                        self.nodes[w].parent = v

        # 3. Lengauer-Tarjan算法核心部分
        for w in range(len(self.nodes)-1, 0, -1):
            p = self.nodes[w].parent
            
            # 计算semi支配节点
            for v_vert in self.nodes[w].hierVertex_.preds:
                if v_vert in self.vertIdx:
                    v = self.vertIdx[v_vert]
                    u = self._eval(v)
                    if self.nodes[w].semi > self.nodes[u].semi:
                        self.nodes[w].semi = self.nodes[u].semi
            
            # 添加到bucket
            AddUnique(self.nodes[self.nodes[w].semi].bucket, w)
            self._link(p, w)

            # 隐式定义即时支配者 Implicitly define immediate dominators
            for v in self.nodes[p].bucket:
                u = self._eval(v)
                if self.nodes[u].semi < self.nodes[v].semi:
                    self.nodes[v].idom = u
                else:
                    self.nodes[v].idom = p
            
            self.nodes[p].bucket.clear()

        # 4. 构建DomNode树
        results = [DomNode(node.hierVertex_) for node in self.nodes]
        #print(results)
        for v in range(1, len(self.nodes)):
            if self.nodes[v].idom != self.nodes[v].semi:
                self.nodes[v].idom = self.nodes[self.nodes[v].idom].idom
            d = self.nodes[v].idom
            results[v].parent = results[d]
            results[d].children.append(results[v])

        # 5. 节点编号用于快速支配判断
        numberer = NodeNumberer()
        numberer.Visit(results[0])
        #NodeNumberer.Visit(results[0])

        return results

    def _eval(self, v_idx):
        """The 'eval' function from Lengauer-Tarjan."""
    
        if self.nodes[v_idx].ancestor == DfNode.NONE:
            return v_idx
        else:
            self._compress(v_idx)
            v_best = self.nodes[v_idx].best
            a_idx = self.nodes[v_idx].ancestor
            a_best = self.nodes[a_idx].best
            if self.nodes[a_best].semi < self.nodes[v_best].semi:
                return a_best
            else:
                return v_best

    def _compress(self, v_idx):
        """The 'compress' function from Lengauer-Tarjan."""
        v_node = self.nodes[v_idx]
        a_idx = v_node.ancestor
        if self.nodes[a_idx].ancestor == DfNode.NONE:
            return
        self._compress(a_idx)
        a_best = self.nodes[a_idx].best
        v_best = self.nodes[v_idx].best
        if self.nodes[a_best].semi < self.nodes[v_best].semi:
            self.nodes[v_idx].best = a_best
        self.nodes[v_idx].ancestor = self.nodes[a_idx].ancestor

    def _link(self, v_idx, w_idx):
        """The 'link' function from Lengauer-Tarjan."""
        s_idx = w_idx
        w_semi = self.nodes[w_idx].semi
        w_best = self.nodes[w_idx].best

        while (self.nodes[s_idx].child != DfNode.NONE and
               self.nodes[w_best].semi < self.nodes[self.nodes[self.nodes[s_idx].child].best].semi ):
            # Combine trees
            cs_idx = self.nodes[s_idx].child
            ss = self.nodes[s_idx].size
            scs = self.nodes[cs_idx].size
            ccs_idx = self.nodes[cs_idx].child
            #sccs = self.nodes[ccs_idx].size
            if ccs_idx != DfNode.NONE:
                sccs = self.nodes[ccs_idx].size
            else:
                sccs = 0

            if ss + sccs >= 2 * scs:
                self.nodes[cs_idx].ancestor = s_idx
                self.nodes[s_idx].child = ccs_idx
            else:
                self.nodes[cs_idx].size = ss
                self.nodes[s_idx].ancestor = cs_idx
                s_idx = cs_idx

        self.nodes[s_idx].best = w_best
        if self.nodes[v_idx].size < self.nodes[w_idx].size:
            self.nodes[v_idx].child, s_idx = s_idx, self.nodes[v_idx].child

        self.nodes[v_idx].size += self.nodes[w_idx].size
        while s_idx != DfNode.NONE:
            self.nodes[s_idx].ancestor = v_idx
            s_idx = self.nodes[s_idx].child
   

    def _dfs_traversal(self, root):
        stack = [root]
        visited = set()
        hier_range = []
    
        while stack:
            vert = stack.pop()
            if vert not in visited:
                visited.add(vert)
                hier_range.append(vert)
                # 反向添加后继节点以保持正确DFS顺序
                for succ in reversed(vert.succs):
                    if succ not in visited:
                        stack.append(succ)
        print(hier_range)
        return hier_range
        
        

class NodeNumberer:
    def __init__(self):
        self.number = 0

    def Visit(self, node: DomNode):
        """对支配树节点进行时间戳编号"""
        self._visit_node(node)
        return None

    def _visit_node(self, node: DomNode):
        node.in_num = self.number
        self.number += 1
        for child in node.children:
            self._visit_node(child)
        node.out_num = self.number
        self.number += 1