import argparse
import time

from src.utils.io_utils import load_transactions_csv
from src.utils.rules import generate_rules

from src.apriori.apriori import apriori
from src.apriori.apriori_tid import apriori_tid
from src.fpgrowth.fp_growth import mine_fp_growth
from src.eclat.eclat import eclat
from src.hybrid.trihybrid import trihybrid_ppam
from src.hybrid.a3miner import a3mine

from src.bench.benchmark import benchmark_all, save_results_csv, plot_runtime, plot_itemset_counts

def fp_counts_to_support(fp_counts, n):
    return {frozenset(k): v/n for k, v in fp_counts.items()}

def run_single(algo: str, transactions_list, transactions_set, minsup: float, minconf: float):
    n = len(transactions_list)
    minsup_count = max(1, int(minsup * n + 1e-9))

    if algo == "apriori":
        t0 = time.perf_counter()
        freq = apriori(transactions_set, minsup)
        elapsed = time.perf_counter() - t0
    elif algo == "aprioritid":
        t0 = time.perf_counter()
        freq = apriori_tid(transactions_set, minsup)
        elapsed = time.perf_counter() - t0
    elif algo == "fpgrowth":
        t0 = time.perf_counter()
        counts = mine_fp_growth(transactions_list, minsup_count)
        elapsed = time.perf_counter() - t0
        freq = fp_counts_to_support(counts, n)
    elif algo == "eclat":
        t0 = time.perf_counter()
        freq = eclat(transactions_set, minsup)
        elapsed = time.perf_counter() - t0
    elif algo == "trihybrid":
        t0 = time.perf_counter()
        res = trihybrid_ppam(transactions_list, transactions_set, minsup)
        elapsed = time.perf_counter() - t0
        freq = res.frequent_itemsets
        print("\n[TriHybrid stats]")
        for k, v in res.stats.items():
            print(f"{k}: {v}")

    elif algo == "a3miner":
        t0 = time.perf_counter()
        res = a3mine(transactions_list, transactions_set, minsup)
        elapsed = time.perf_counter() - t0
        freq = res.frequent_itemsets
        print("\n[A3Miner selection]")
        for k, v in res.stats.items():
            print(f"{k}: {v}")

    else:
        raise ValueError("Unknown algo")

    rules = generate_rules(freq, minconf)

    print(f"\nAlgorithm: {algo}")
    print(f"Transactions: {n}")
    print(f"minsup: {minsup} (count >= {minsup_count})")
    print(f"minconf: {minconf}")
    print(f"Runtime: {elapsed:.6f} sec")
    print(f"Frequent itemsets: {len(freq)}")
    print(f"Association rules: {len(rules)}")

    print("\nTop 10 rules:")
    for r in rules[:10]:
        lift = r["lift"]
        lift_str = f"{lift:.3f}" if lift is not None else "NA"
        print(f"{set(r['antecedent'])} -> {set(r['consequent'])} | "
              f"support={r['support']:.3f}, conf={r['confidence']:.3f}, lift={lift_str}")

def main():
    parser = argparse.ArgumentParser(description="Association Mining: Apriori, FP-growth, Eclat, and TriHybrid-PPAM")
    parser.add_argument("--dataset", type=str, default="data/market_basket_50.csv")
    parser.add_argument("--algo", type=str, default="trihybrid",
                        choices=["apriori", "aprioritid", "fpgrowth", "eclat", "trihybrid", "a3miner", "benchmark"])
    parser.add_argument("--minsup", type=float, default=0.30)
    parser.add_argument("--minconf", type=float, default=0.70)
    parser.add_argument("--plot", action="store_true", help="Generate plots (benchmark mode)")
    args = parser.parse_args()

    tx_list, tx_set = load_transactions_csv(args.dataset)

    if args.algo == "benchmark":
        minsups = [0.50, 0.40, 0.30, 0.25, 0.20]
        results = benchmark_all(tx_list, tx_set, minsups)
        save_results_csv(results, "benchmark_results.csv")
        print("Saved: benchmark_results.csv")
        if args.plot:
            plot_runtime(results, "support_vs_runtime_all.png")
            plot_itemset_counts(results, "support_vs_itemsets_all.png")
            print("Saved: support_vs_runtime_all.png")
            print("Saved: support_vs_itemsets_all.png")
        return

    run_single(args.algo, tx_list, tx_set, args.minsup, args.minconf)

if __name__ == "__main__":
    main()
