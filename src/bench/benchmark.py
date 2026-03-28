from __future__ import annotations
import time
from typing import Dict, List, Any
import csv

import matplotlib.pyplot as plt

from src.apriori.apriori import apriori
from src.apriori.apriori_tid import apriori_tid
from src.fpgrowth.fp_growth import mine_fp_growth
from src.eclat.eclat import eclat
from src.hybrid.trihybrid import trihybrid_ppam
from src.hybrid.a3miner import a3mine

def _fp_counts_to_frac(counts: Dict[tuple, int], n: int) -> Dict[frozenset, float]:
    return {frozenset(k): v / n for k, v in counts.items()}

def benchmark_all(transactions_list, transactions_set, minsups: List[float], density_threshold: float = 2.6) -> Dict[str, Any]:
    n = len(transactions_list)
    rows = []

    for minsup in minsups:
        minsup_count = max(1, int(minsup * n + 1e-9))

        # Apriori
        t0 = time.perf_counter()
        freq_ap = apriori(transactions_set, minsup)
        t_ap = time.perf_counter() - t0

        # AprioriTid
        t0 = time.perf_counter()
        freq_at = apriori_tid(transactions_set, minsup)
        t_at = time.perf_counter() - t0

        # FP-growth
        t0 = time.perf_counter()
        fp_counts = mine_fp_growth(transactions_list, minsup_count)
        t_fp = time.perf_counter() - t0
        freq_fp = _fp_counts_to_frac(fp_counts, n)

        # Eclat
        t0 = time.perf_counter()
        freq_ec = eclat(transactions_set, minsup)
        t_ec = time.perf_counter() - t0

        # TriHybrid
        t0 = time.perf_counter()
        tri = trihybrid_ppam(transactions_list, transactions_set, minsup, density_threshold=density_threshold)
        t_th = time.perf_counter() - t0
        freq_th = tri.frequent_itemsets

        # A3Miner (adaptive selector)
        t0 = time.perf_counter()
        a3 = a3mine(transactions_list, transactions_set, minsup)
        t_a3 = time.perf_counter() - t0
        freq_a3 = a3.frequent_itemsets

        rows.append({
            "minsup": minsup,
            "minsup_count": minsup_count,
            "apriori_s": t_ap,
            "aprioriTid_s": t_at,
            "fpgrowth_s": t_fp,
            "eclat_s": t_ec,
            "trihybrid_s": t_th,
            "a3miner_s": t_a3,
            "n_itemsets_apriori": len(freq_ap),
            "n_itemsets_fpgrowth": len(freq_fp),
            "n_itemsets_eclat": len(freq_ec),
            "n_itemsets_trihybrid": len(freq_th),
            "n_itemsets_a3miner": len(freq_a3),
            "a3miner_engine": a3.chosen_engine,
            "trihybrid_pairs_checked": tri.stats.get("pairs_checked"),
            "trihybrid_freq_pairs": tri.stats.get("freq_pairs_found"),
            "trihybrid_projections": tri.stats.get("projections_mined"),
            "trihybrid_fp_proj": tri.stats.get("mined_with_fp"),
            "trihybrid_eclat_proj": tri.stats.get("mined_with_eclat"),
        })

    return {"n": n, "rows": rows}

def save_results_csv(results: Dict[str, Any], path: str) -> None:
    rows = results["rows"]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def plot_runtime(results: Dict[str, Any], out_path: str) -> None:
    rows = results["rows"]
    minsups = [r["minsup"] for r in rows]

    plt.figure()
    plt.plot(minsups, [r["apriori_s"] for r in rows], label="Apriori")
    plt.plot(minsups, [r["aprioriTid_s"] for r in rows], label="AprioriTid")
    plt.plot(minsups, [r["fpgrowth_s"] for r in rows], label="FP-growth")
    plt.plot(minsups, [r["eclat_s"] for r in rows], label="Eclat")
    plt.plot(minsups, [r["trihybrid_s"] for r in rows], label="TriHybrid-PPAM")
    plt.plot(minsups, [r["a3miner_s"] for r in rows], label="A3Miner (adaptive)")
    plt.xlabel("Minimum Support")
    plt.ylabel("Runtime (seconds)")
    plt.title("Support vs Runtime (All Algorithms)")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

def plot_itemset_counts(results: Dict[str, Any], out_path: str) -> None:
    rows = results["rows"]
    minsups = [r["minsup"] for r in rows]

    plt.figure()
    plt.plot(minsups, [r["n_itemsets_apriori"] for r in rows], label="Apriori")
    plt.plot(minsups, [r["n_itemsets_fpgrowth"] for r in rows], label="FP-growth")
    plt.plot(minsups, [r["n_itemsets_eclat"] for r in rows], label="Eclat")
    plt.plot(minsups, [r["n_itemsets_trihybrid"] for r in rows], label="TriHybrid-PPAM")
    plt.plot(minsups, [r["n_itemsets_a3miner"] for r in rows], label="A3Miner")
    plt.xlabel("Minimum Support")
    plt.ylabel("# Frequent itemsets")
    plt.title("Support vs #Frequent Itemsets")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
