from abc import ABC, abstractmethod
from weakref import ref, WeakKeyDictionary
import sys

class DomNode:
    def __init__(self, vertex):
        self.vertex = vertex  # 原始顶点的强引用
        self.parent = None    # 支配树父节点
        self.children = []    # 支配树子节点列表
        self.in_ = 0          # DFS进入时间戳
        self.out = 0          # DFS离开时间戳

    def Dominates(self, other, strict=False):
        """判断当前节点是否支配other节点"""
        if strict:
            return self.in_ < other.in_ and self.out > other.out
        else:
            return self.in_ <= other.in_ and self.out >= other.out

class DomTreeVisitor(ABC):
    @abstractmethod
    def Visit(self, node, *args):
        pass

class DfNode:
    NONE = -1  # 特殊标记值

    def __init__(self, vertex):
        self.vertex = vertex      # 对应原始顶点
        self.parent = self.NONE   # DFS树父节点索引
        self.semi = self.NONE     # 半支配编号
        self.bucket = []          # 半支配桶
        self.idom = self.NONE     # 即时支配者索引
        self.ancestor = self.NONE # 并查集祖先
        self.best = 0             # 最优路径节点
        self.size = 0             # 子树大小
        self.child = self.NONE    # 孩子索引

class DomBuilder:
    def __init__(self, getPreds=None, getSuccs=None):
        """初始化构建器
        Args:
            getPreds: 获取前驱节点的函数
            getSuccs: 获取后继节点的函数
        """
        self.getPreds = getPreds or (lambda vert: vert.Preds())
        self.getSuccs = getSuccs or (lambda vert: vert.Succs())
        self.nodes = []           # DfNode数组
        self.vertIdx = {}         # 顶点到索引的映射

    def Build(self, root):
        """构建支配树"""
        assert root is not None
        
        # 1. 深度优先遍历收集所有节点
        self.nodes.clear()
        self.vertIdx.clear()
        count = 0
        
        for vertex in self._dfs_traversal(root):
            df_node = DfNode(vertex)
            df_node.parent = DfNode.NONE
            df_node.semi = DfNode.NONE
            df_node.idom = DfNode.NONE
            df_node.ancestor = DfNode.NONE
            df_node.best = count
            df_node.size = 0
            self.nodes.append(df_node)
            self.vertIdx[vertex] = count
            count += 1

        if len(self.nodes) <= 1:
            print("Graph is trivial. No need to build dominator tree.")
            return []

        # 2. 初始化semi和父节点关系
        for v in range(len(self.nodes)):
            self.nodes[v].semi = v
            # 设置DFS树的父节点
            for w_vert in self.getSuccs(self.nodes[v].vertex):
                if w_vert in self.vertIdx:
                    w = self.vertIdx[w_vert]
                    if self.nodes[w].semi == DfNode.NONE:
                        self.nodes[w].parent = v

        # 3. Lengauer-Tarjan算法核心部分
        for w in range(len(self.nodes)-1, 0, -1):
            p = self.nodes[w].parent
            
            # 计算semi支配节点
            for v_vert in self.getPreds(self.nodes[w].vertex):
                if v_vert in self.vertIdx:
                    v = self.vertIdx[v_vert]
                    u = self._eval(v)
                    if self.nodes[w].semi > self.nodes[u].semi:
                        self.nodes[w].semi = self.nodes[u].semi
            
            # 添加到bucket
            if self.nodes[self.nodes[w].semi].bucket is None:
                self.nodes[self.nodes[w].semi].bucket = []
            self.nodes[self.nodes[w].semi].bucket.append(w)
            
            self._link(p, w)

            # 隐式定义即时支配者
            for v in self.nodes[p].bucket:
                u = self._eval(v)
                if self.nodes[u].semi < self.nodes[v].semi:
                    self.nodes[v].idom = u
                else:
                    self.nodes[v].idom = p
            
            self.nodes[p].bucket.clear()

        # 4. 构建DomNode树
        results = [DomNode(node.vertex) for node in self.nodes]
        for v in range(1, len(self.nodes)):
            if self.nodes[v].idom != self.nodes[v].semi:
                self.nodes[v].idom = self.nodes[self.nodes[v].idom].idom
            d = self.nodes[v].idom
            results[v].parent = results[d]
            results[d].children.append(results[v])

        # 5. 节点编号用于快速支配判断
        numberer = NodeNumberer()
        numberer.Visit(results[0])

        return results

    def _dfs_traversal(self, root):
        """深度优先遍历生成器"""
        visited = set()
        stack = [root]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            yield vertex
            
            # 添加未访问的后继节点
            for succ in reversed(self.getSuccs(vertex)):
                if succ not in visited:
                    stack.append(succ)

    def _eval(self, v):
        if self.nodes[v].ancestor == DfNode.NONE:
            return v
        else:
            self._compress(v)
            b = self.nodes[v].best
            a = self.nodes[v].ancestor
            ba = self.nodes[a].best
            return ba if self.nodes[ba].semi < self.nodes[b].semi else b

    def _compress(self, v):
        a = self.nodes[v].ancestor
        if self.nodes[a].ancestor == DfNode.NONE:
            return
        self._compress(a)
        if self.nodes[self.nodes[a].best].semi < self.nodes[self.nodes[v].best].semi:
            self.nodes[v].best = self.nodes[a].best
        self.nodes[v].ancestor = self.nodes[a].ancestor

    def _link(self, v, w):
        s = w
        while self.nodes[s].child != DfNode.NONE and \
              self.nodes[self.nodes[s].child].best < self.nodes[s].best:
            # 合并子链中的前两个树
            cs = self.nodes[s].child
            ss = self.nodes[s].size
            ccs = self.nodes[cs].child
            scs = self.nodes[cs].size
            
            if ss + self._get_size(ccs) >= 2 * scs:
                self.nodes[cs].ancestor = s
                self.nodes[s].child = ccs
            else:
                self.nodes[cs].size = ss
                self.nodes[s].ancestor = cs
                s = cs

        # 合并森林
        self.nodes[s].best = self.nodes[w].best
        if self._get_size(v) < self._get_size(w):
            s, child_v = self._swap_and_get_child(s, self.nodes[v].child)
            self.nodes[v].child = child_v
        
        self.nodes[v].size += self.nodes[w].size
        
        while s != DfNode.NONE:
            self.nodes[s].ancestor = v
            s = self.nodes[s].child

    def _get_size(self, index):
        return self.nodes[index].size if index != DfNode.NONE else 0

    def _swap_and_get_child(self, a, b):
        # 辅助方法实现交换逻辑
        return b, a

class NodeNumberer:
    def Visit(self, node):
        """对支配树节点进行时间戳编号"""
        self.number = 0
        self._visit_node(node)
        return None

    def _visit_node(self, node):
        node.in_ = self.number
        self.number += 1
        for child in node.children:
            self._visit_node(child)
        node.out = self.number
        self.number += 1