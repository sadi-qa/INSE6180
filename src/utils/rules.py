from itertools import combinations
from typing import Dict, FrozenSet, List, Optional

Itemset = FrozenSet[str]

def generate_rules(
    frequent_itemsets: Dict[Itemset, float],
    minconf: float
) -> List[dict]:
    """
    Generate association rules from frequent itemsets.

    frequent_itemsets: dict itemset -> support fraction in [0,1]
    minconf: minimum confidence threshold in [0,1]

    Returns list of dicts: antecedent, consequent, support, confidence, lift
    """
    rules: List[dict] = []

    for itemset, sup_xy in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                X = frozenset(antecedent_tuple)
                Y = itemset - X
                if not Y:
                    continue

                sup_x = frequent_itemsets.get(X)
                sup_y = frequent_itemsets.get(Y)
                if sup_x is None or sup_x == 0:
                    continue

                conf = sup_xy / sup_x
                if conf >= minconf:
                    lift: Optional[float] = None
                    if sup_y is not None and sup_y > 0:
                        lift = conf / sup_y

                    rules.append({
                        "antecedent": X,
                        "consequent": Y,
                        "support": sup_xy,
                        "confidence": conf,
                        "lift": lift
                    })

    rules.sort(key=lambda d: (d["confidence"], d["support"]), reverse=True)
    return rules
