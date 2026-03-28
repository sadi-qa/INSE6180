from __future__ import annotations
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

from src.apriori.hashtree import HashTree

Itemset = FrozenSet[str]

def find_frequent_1_itemsets(transactions: List[Set[str]], minsup: float) -> Dict[Itemset, float]:
    counts = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[item] += 1
    n = len(transactions)
    out: Dict[Itemset, float] = {}
    for item, ct in counts.items():
        sup = ct / n
        if sup >= minsup:
            out[frozenset([item])] = sup
    return out

def apriori_gen(L_prev: Dict[Itemset, float], k: int) -> Set[Itemset]:
    # join step
    L_list = sorted([tuple(sorted(s)) for s in L_prev.keys()])
    candidates: Set[Itemset] = set()
    for i in range(len(L_list)):
        for j in range(i+1, len(L_list)):
            a, b = L_list[i], L_list[j]
            if a[:k-2] == b[:k-2]:
                candidates.add(frozenset(a) | frozenset(b))
            else:
                # because sorted lexicographically, once prefix differs we can break
                # but only if we group by prefix; keep simple and continue
                pass
    return candidates

def prune_candidates(candidates: Set[Itemset], L_prev: Dict[Itemset, float]) -> Set[Itemset]:
    pruned: Set[Itemset] = set()
    for c in candidates:
        ok = True
        for item in c:
            if (c - frozenset([item])) not in L_prev:
                ok = False
                break
        if ok:
            pruned.add(c)
    return pruned

def count_candidates_hashtree(candidates: Set[Itemset], transactions: List[Set[str]], minsup: float, k: int) -> Dict[Itemset, float]:
    tree = HashTree(k=k, bucket_size=7, leaf_threshold=50)
    for c in candidates:
        tree.insert(c)
    counts = defaultdict(int)
    for t in transactions:
        for c in tree.find_candidates(t):
            counts[c] += 1
    n = len(transactions)
    Lk: Dict[Itemset, float] = {}
    for c, ct in counts.items():
        sup = ct / n
        if sup >= minsup:
            Lk[c] = sup
    return Lk

def apriori(transactions: List[Set[str]], minsup: float) -> Dict[Itemset, float]:
    """
    Apriori frequent itemset mining (Paper 1), using:
      - join + prune candidate generation
      - hash-tree based candidate counting (subset function)
    Returns: dict itemset -> support fraction
    """
    all_freq: Dict[Itemset, float] = {}
    L_prev = find_frequent_1_itemsets(transactions, minsup)
    all_freq.update(L_prev)

    k = 2
    while L_prev:
        Ck = apriori_gen(L_prev, k)
        Ck = prune_candidates(Ck, L_prev)
        if not Ck:
            break
        Lk = count_candidates_hashtree(Ck, transactions, minsup, k)
        if not Lk:
            break
        all_freq.update(Lk)
        L_prev = Lk
        k += 1

    return all_freq
