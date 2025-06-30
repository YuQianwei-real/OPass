import weakref
from typing import List, Set, Optional, Callable


class VertexBase:
    def __init__(self):
        self._preds: List[weakref.ref] = []
        self._succs: List['VertexBase'] = []

    @property
    def preds(self) -> List['VertexBase']:
        return [p() for p in self._preds if p() is not None]

    @property
    def succs(self) -> List['VertexBase']:
        return self._succs

    def add_pred(self, pred: 'VertexBase'):
        wp = weakref.ref(pred)
        if wp not in self._preds:
            self._preds.append(wp)

    def add_succ(self, succ: 'VertexBase'):
        if succ not in self._succs:
            self._succs.append(succ)

    def remove_pred(self, pred: 'VertexBase'):
        wp = weakref.ref(pred)
        self._preds = [p for p in self._preds if p != wp]

    def remove_succ(self, succ: 'VertexBase'):
        self._succs = [s for s in self._succs if s is not succ]

    @classmethod
    def connect(cls, tail: 'VertexBase', head: 'VertexBase'):
        tail.add_succ(head)
        head.add_pred(tail)

    @classmethod
    def disconnect(cls, tail: 'VertexBase', head: 'VertexBase'):
        tail.remove_succ(head)
        head.remove_pred(tail)

    @classmethod
    def replace(cls, old_vert: 'VertexBase', new_vert: 'VertexBase'):
        for pred in old_vert.preds:
            pred.replace_succ(old_vert, new_vert)
        for succ in old_vert.succs:
            succ.replace_pred(old_vert, new_vert)

    def replace_succ(self, old_succ: 'VertexBase', new_succ: 'VertexBase'):
        if new_succ in self._succs:
            self._succs.remove(old_succ)
        else:
            try:
                idx = self._succs.index(old_succ)
                self._succs[idx] = new_succ
            except ValueError:
                pass

    def replace_pred(self, old_pred: 'VertexBase', new_pred: 'VertexBase'):
        wp_old = weakref.ref(old_pred)
        wp_new = weakref.ref(new_pred)
        if wp_new in self._preds:
            self._preds = [p for p in self._preds if p != wp_old]
        else:
            try:
                idx = self._preds.index(wp_old)
                self._preds[idx] = wp_new
            except ValueError:
                pass


class DfsIter:
    def __init__(self, inputs, get_succs: Callable = lambda v: v.succs):
        self.stack = list(reversed(inputs))
        self.get_succs = get_succs
        self.visited: Set['VertexBase'] = set()

    def __iter__(self):
        return self

    def __next__(self) -> Optional['VertexBase']:
        while self.stack:
            vertex = self.stack.pop()
            if vertex in self.visited:
                continue
            self.visited.add(vertex)
            self.stack.extend(reversed(self.get_succs(vertex)))
            return vertex
        raise StopIteration

class RpoIter:
    def __init__(self, outputs, get_preds: Callable = lambda v: v.preds):
        self.stack = [(v, False) for v in reversed(outputs)]
        self.get_preds = get_preds
        self.visited: Set['VertexBase'] = set()

    def __iter__(self):
        return self

    def __next__(self) -> Optional['VertexBase']:
        while self.stack:
            vertex, visited = self.stack.pop()
            if vertex in self.visited:
                continue
            if visited:
                self.visited.add(vertex)
                return vertex
            self.stack.append((vertex, True))
            self.stack.extend([(p, False) for p in reversed(self.get_preds(vertex))])
        raise StopIteration

class VertRange:
    def __init__(self, vertices):
        self.vertices = vertices

    def __iter__(self):
        return iter(self.vertices)

    def dfs(self):
        return DfsIter(self.vertices)

    def rpo(self):
        return RpoIter(self.vertices)