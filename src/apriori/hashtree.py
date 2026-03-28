from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, FrozenSet, Set, Optional

Itemset = FrozenSet[str]

def _hash(item: str, mod: int) -> int:
    return (hash(item) & 0x7fffffff) % mod

@dataclass
class _Node:
    depth: int
    is_leaf: bool = True
    candidates: List[Itemset] = field(default_factory=list)
    children: Dict[int, "_Node"] = field(default_factory=dict)

class HashTree:
    """
    A practical hash-tree for candidate counting inspired by Apriori's subset function.
    The paper describes storing candidates in a hash-tree and retrieving candidates
    contained in a transaction efficiently. (Paper 1, Section 2.1.2)
    """
    def __init__(self, k: int, bucket_size: int = 7, leaf_threshold: int = 50):
        self.k = k
        self.bucket_size = bucket_size
        self.leaf_threshold = leaf_threshold
        self.root = _Node(depth=1)

    def insert(self, itemset: Itemset) -> None:
        items = sorted(itemset)
        node = self.root

        while True:
            if node.is_leaf:
                node.candidates.append(itemset)
                if len(node.candidates) > self.leaf_threshold and node.depth < self.k:
                    self._split(node)
                return
            idx = node.depth - 1
            h = _hash(items[idx], self.bucket_size)
            if h not in node.children:
                node.children[h] = _Node(depth=node.depth + 1)
            node = node.children[h]

    def _split(self, node: _Node) -> None:
        old = node.candidates
        node.is_leaf = False
        node.candidates = []
        node.children = {}

        for cand in old:
            items = sorted(cand)
            idx = node.depth - 1
            h = _hash(items[idx], self.bucket_size)
            if h not in node.children:
                node.children[h] = _Node(depth=node.depth + 1)
            node.children[h].candidates.append(cand)

    def find_candidates(self, transaction: Set[str]) -> List[Itemset]:
        t_items = sorted(transaction)
        out: List[Itemset] = []
        self._subset(self.root, t_items, start=0, transaction=transaction, out=out)
        return out

    def _subset(self, node: _Node, t_items: List[str], start: int, transaction: Set[str], out: List[Itemset]) -> None:
        if node.is_leaf:
            for cand in node.candidates:
                if cand.issubset(transaction):
                    out.append(cand)
            return

        depth_idx = node.depth - 1
        for i in range(start, len(t_items)):
            if depth_idx >= len(t_items):
                break
            h = _hash(t_items[i], self.bucket_size)
            child = node.children.get(h)
            if child is not None:
                self._subset(child, t_items, start=i+1, transaction=transaction, out=out)
