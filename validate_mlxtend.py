import argparse
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as mlx_apriori
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

from src.utils.io_utils import load_transactions_csv
from src.apriori.apriori import apriori as my_apriori
from src.fpgrowth.fp_growth import mine_fp_growth
from src.eclat.eclat import eclat as my_eclat
from src.hybrid.a3miner import a3mine

def to_onehot(transactions_list):
    te = TransactionEncoder()
    arr = te.fit(transactions_list).transform(transactions_list)
    df = pd.DataFrame(arr, columns=te.columns_)
    return df

def itemsets_from_mlxtend(df_out):
    return {frozenset(row["itemsets"]): float(row["support"]) for _, row in df_out.iterrows()}

def compare(a, b, tol=1e-9):
    """
    Compare two dicts itemset->support.
    Returns (only_in_a, only_in_b, mismatched_supports)
    """
    only_a = [k for k in a.keys() if k not in b]
    only_b = [k for k in b.keys() if k not in a]
    mism = []
    for k in a.keys():
        if k in b and abs(a[k] - b[k]) > tol:
            mism.append((k, a[k], b[k]))
    return only_a, only_b, mism

def main():
    p = argparse.ArgumentParser(description="Validate implementations against mlxtend")
    p.add_argument("--dataset", type=str, default="data/market_basket_50.csv")
    p.add_argument("--minsup", type=float, default=0.3)
    p.add_argument("--engine", type=str, default="apriori", choices=["apriori","fpgrowth","eclat","a3miner"])
    args = p.parse_args()

    tx_list, tx_set = load_transactions_csv(args.dataset)
    n = len(tx_list)
    minsup_count = max(1, int(args.minsup * n + 1e-9))

    df_onehot = to_onehot(tx_list)

    # mlxtend baselines
    mlx_ap = itemsets_from_mlxtend(mlx_apriori(df_onehot, min_support=args.minsup, use_colnames=True))
    mlx_fp = itemsets_from_mlxtend(mlx_fpgrowth(df_onehot, min_support=args.minsup, use_colnames=True))

    # our outputs
    if args.engine == "apriori":
        mine = my_apriori(tx_set, args.minsup)
        baseline = mlx_ap
        baseline_name = "mlxtend.apriori"
    elif args.engine == "fpgrowth":
        counts = mine_fp_growth(tx_list, minsup_count)
        mine = {frozenset(k): v/n for k, v in counts.items()}
        baseline = mlx_fp
        baseline_name = "mlxtend.fpgrowth"
    elif args.engine == "eclat":
        mine = my_eclat(tx_set, args.minsup)
        baseline = mlx_ap  # both should match frequent itemsets set
        baseline_name = "mlxtend.apriori"
    else:  # a3miner
        res = a3mine(tx_list, tx_set, args.minsup)
        mine = res.frequent_itemsets
        baseline = mlx_ap
        baseline_name = "mlxtend.apriori"

    only_mine, only_base, mism = compare(mine, baseline, tol=1e-9)

    print(f"Validation engine: {args.engine}")
    print(f"Baseline: {baseline_name}")
    print(f"minsup={args.minsup}")
    print(f"Our itemsets: {len(mine)} | Baseline itemsets: {len(baseline)}")
    print(f"Only in ours: {len(only_mine)}")
    print(f"Only in baseline: {len(only_base)}")
    print(f"Support mismatches: {len(mism)}")

    if only_mine[:10]:
        print("\nExamples only in ours:", [set(x) for x in only_mine[:10]])
    if only_base[:10]:
        print("\nExamples only in baseline:", [set(x) for x in only_base[:10]])
    if mism[:10]:
        print("\nExamples mismatches:", [(set(k), a, b) for k, a, b in mism[:10]])

    if len(only_mine)==0 and len(only_base)==0 and len(mism)==0:
        print("\n✅ Validation PASSED (exact match).")
    else:
        print("\n⚠️ Validation found differences. Investigate thresholds / rounding / parsing.")

if __name__ == "__main__":
    main()
