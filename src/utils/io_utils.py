import pandas as pd
from typing import List, Set, Tuple

def load_transactions_csv(path: str, items_col: str = "items", sep: str = ",") -> Tuple[List[List[str]], List[Set[str]]]:
    """
    Load a CSV where each row represents a transaction.
    Expected columns: transaction_id, items
    items is a comma-separated string.

    Returns:
      - transactions_list: List[List[str]] (for FP-growth style code)
      - transactions_set:  List[Set[str]]  (for Apriori/Eclat style code)
    """
    df = pd.read_csv(path)
    transactions_list: List[List[str]] = []
    transactions_set: List[Set[str]] = []

    for items in df[items_col].astype(str).tolist():
        tx = [i.strip() for i in items.split(sep) if i.strip()]
        transactions_list.append(tx)
        transactions_set.append(set(tx))

    return transactions_list, transactions_set
