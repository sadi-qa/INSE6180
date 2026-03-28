from __future__ import annotations
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from src.fpgrowth.fp_tree import FPTree, FPNode

Pattern = Tuple[str, ...]  # sorted tuple
WeightedTransaction = Tuple[List[str], int]  # (items, count)

def build_fp_tree(transactions: List[List[str]], minsup_count: int, weights: Optional[List[int]] = None) -> Tuple[Optional[FPTree], Dict[str, int]]:
    """
    Build FP-tree with 2-pass style counting. For conditional trees, you can supply weights.
    """
    item_counter = Counter()
    if weights is None:
        for t in transactions:
            item_counter.update(t)
    else:
        for t, w in zip(transactions, weights):
            for item in t:
                item_counter[item] += w

    freq = {i: c for i, c in item_counter.items() if c >= minsup_count}
    if not freq:
        return None, {}

    order = [i for i, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]
    rank = {item: idx for idx, item in enumerate(order)}

    tree = FPTree()

    if weights is None:
        for t in transactions:
            filtered = [i for i in t if i in freq]
            filtered.sort(key=lambda x: rank[x])
            if filtered:
                tree.add_transaction(filtered, 1)
    else:
        for t, w in zip(transactions, weights):
            filtered = [i for i in t if i in freq]
            filtered.sort(key=lambda x: rank[x])
            if filtered:
                tree.add_transaction(filtered, w)

    tree.set_supports(freq)
    return tree, freq

def _ascend_prefix_path(node: FPNode) -> List[str]:
    path: List[str] = []
    cur = node.parent
    while cur is not None and cur.item is not None:
        path.append(cur.item)
        cur = cur.parent
    path.reverse()
    return path

def conditional_pattern_base(tree: FPTree, item: str) -> List[Tuple[List[str], int]]:
    """
    Return list of (prefix_path, count) pairs for the given suffix item.
    """
    base: List[Tuple[List[str], int]] = []
    _, head = tree.header_table[item]
    node = head
    while node is not None:
        prefix = _ascend_prefix_path(node)
        if prefix:
            base.append((prefix, node.count))
        node = node.node_link
    return base

def mine_fp_growth(transactions: List[List[str]], minsup_count: int) -> Dict[Pattern, int]:
    tree, _ = build_fp_tree(transactions, minsup_count)
    if tree is None:
        return {}
    patterns: Dict[Pattern, int] = {}
    _mine_tree(tree, minsup_count, suffix=tuple(), patterns=patterns)
    return patterns

def _mine_tree(tree: FPTree, minsup_count: int, suffix: Tuple[str, ...], patterns: Dict[Pattern, int]) -> None:
    if tree.is_single_path():
        nodes = tree.single_path_nodes()
        items_counts = [(n.item, n.count) for n in nodes if n.item is not None]
        items = [i for i, _ in items_counts]

        for r in range(1, len(items) + 1):
            for combo in combinations(items, r):
                sup = min(next(c for (i, c) in items_counts if i == x) for x in combo)
                pat = tuple(sorted(combo + suffix))
                patterns[pat] = max(patterns.get(pat, 0), sup)
        return

    for item in tree.header_items_least_frequent_first():
        support, _ = tree.header_table[item]
        pat = tuple(sorted((item,) + suffix))
        patterns[pat] = max(patterns.get(pat, 0), support)

        base = conditional_pattern_base(tree, item)
        if not base:
            continue

        cond_transactions = [p for (p, _) in base]
        cond_weights = [w for (_, w) in base]

        cond_tree, _ = build_fp_tree(cond_transactions, minsup_count, weights=cond_weights)
        if cond_tree is not None:
            _mine_tree(cond_tree, minsup_count, suffix=(item,) + suffix, patterns=patterns)
