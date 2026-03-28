from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class FPNode:
    item: Optional[str]
    count: int
    parent: Optional["FPNode"]
    children: Dict[str, "FPNode"]
    node_link: Optional["FPNode"]

    def __init__(self, item: Optional[str], parent: Optional["FPNode"]):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}
        self.node_link = None

class FPTree:
    """
    FP-tree structure (Paper 2): root + header table with node-links.
    header_table[item] = (support_count, head_node)
    """
    def __init__(self):
        self.root = FPNode(item=None, parent=None)
        self.header_table: Dict[str, Tuple[int, Optional[FPNode]]] = {}

    def add_transaction(self, items: List[str], count: int = 1) -> None:
        cur = self.root
        for item in items:
            if item in cur.children:
                child = cur.children[item]
                child.count += count
            else:
                child = FPNode(item=item, parent=cur)
                child.count = count
                cur.children[item] = child
                self._link_in_header(item, child)
            cur = child

    def _link_in_header(self, item: str, node: FPNode) -> None:
        if item not in self.header_table:
            self.header_table[item] = (0, node)
            return
        _, head = self.header_table[item]
        assert head is not None
        cur = head
        while cur.node_link is not None:
            cur = cur.node_link
        cur.node_link = node

    def set_supports(self, supports: Dict[str, int]) -> None:
        for item, sup in supports.items():
            head = self.header_table[item][1] if item in self.header_table else None
            self.header_table[item] = (sup, head)

    def is_single_path(self) -> bool:
        cur = self.root
        while True:
            if len(cur.children) == 0:
                return True
            if len(cur.children) > 1:
                return False
            cur = next(iter(cur.children.values()))

    def single_path_nodes(self) -> List[FPNode]:
        nodes: List[FPNode] = []
        cur = self.root
        while len(cur.children) == 1:
            cur = next(iter(cur.children.values()))
            nodes.append(cur)
        return nodes

    def header_items_least_frequent_first(self) -> List[str]:
        return [item for item, _ in sorted(self.header_table.items(), key=lambda x: x[1][0])]
