from ..graph.base import Vertex

from typing import List, Dict, Set, Optional, Tuple, Any



def replace_succ_of_pred(pred:Vertex, old_vert:Vertex, new_vert:Vertex):
    if new_vert in pred.succs:
        pred.succs = [v for v in pred.succs if v != old_vert]
    else:
        pred.succs = [new_vert if v == old_vert else v for v in pred.succs]
    return

def replace_succ_of_all_preds(old_vert:'Vertex', new_vert:'Vertex'):
    #"""替换所有前驱节点的后继节点"""
    for pred in old_vert.preds: # 解析弱引用
        replace_succ_of_pred(pred, old_vert, new_vert)
    return

def replace_pred_of_succ(succ: Vertex, old_vert:Vertex, new_vert:Vertex):
    #替换后继节点succ的前驱节点中的old_vert为new_vert
    if new_vert in succ.preds:
        # 移除所有指向old_vert的弱引用
        succ.preds = [v for v in succ.preds if v != old_vert]
    else:
        # 替换指向old_vert的弱引用
        succ.preds = [new_vert if v == old_vert else v for v in succ.preds]
    return
    
def replace_pred_of_all_succs(old_vert: Vertex, new_vert: Vertex):
    #"""替换所有后继节点的前驱节点"""
    for succ in old_vert.succs:
        replace_pred_of_succ(succ, old_vert, new_vert)
    return

def AddUnique(lst: List[Any], item: Any) -> bool:
    """添加唯一元素"""
    if item not in lst:
        lst.append(item)
        return True
    return False

def Remove(lst: List[Any], item: Any) -> bool:
    """移除指定元素"""
    try:
        lst.remove(item)
        return True
    except ValueError:
        return False