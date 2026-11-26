---
title: "LeetCode 146 — LRU Cache: O(1) Design & Think-Aloud Interview Notes"
date: 2026-11-12
category: algorithms
summary: "Interview-ready English breakdown of LRU Cache: why combining a hash map with a doubly linked list achieves strict O(1) time, plus the dummy-node trick that eliminates pointer bugs."
tags: [leetcode, data-structures, design, think-aloud, python]
featured: true
---

> **Goal of this series**: Every algorithm note follows the four-step structure I use to practice *think-aloud* technical interviews in English — **Intuition → Approach & Complexity → Edge Cases → Clean Code**.

## 1. Intuition

An LRU (Least Recently Used) Cache needs two operations in **$O(1)$ average time**:
- `get(key)`: return the value if present, and mark this key as **most recently used**.
- `put(key, value)`: insert or update the key, mark it as most recently used, and evict the **least recently used** key if capacity is exceeded.

Think about what each data structure gives us:
- A **Hash Map** gives $O(1)$ lookup by `key`, but does not track recency order.
- A **Doubly Linked List** gives $O(1)$ insertion at the head and $O(1)$ deletion of an arbitrary node (given a pointer to it), but $O(n)$ lookup.

Combine them: the hash map stores `key -> Node`, while the doubly linked list orders nodes from **most recently used** (front) to **least recently used** (back).

## 2. Approach & Complexity

To eliminate `None` pointer checks during insertion/removal, keep two sentinel (dummy) nodes: `head` and `tail`, initialized as `head <-> tail`.

| Operation | Steps | Time |
| :--- | :--- | :--- |
| `get(key)` | Lookup in `cache`; if found, `_remove(node)` and `_insert_front(node)` | $O(1)$ |
| `put(key, val)` | If key exists, remove old node; insert new node at front; if `len > cap`, evict `tail.prev` | $O(1)$ |

- **Time Complexity**: $O(1)$ per `get` and `put`.
- **Space Complexity**: $O(\text{capacity})$ for storing up to `capacity` nodes in both the map and the list.

## 3. Edge Cases to Mention Out Loud

1. **Updating an existing key when cache is full**: Must update its value and move it to the front **without** evicting another entry.
2. **Capacity = 1**: Inserting a second distinct key immediately evicts the first; sentinel `head`/`tail` pointers handle this with zero special branches.
3. **Key not found in `get`**: Return `-1` without mutating list order.

## 4. Clean Implementation (Python)

```python
class Node:
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev: Node | None = None
        self.next: Node | None = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[int, Node] = {}

        # Sentinel nodes: head.next is MRU, tail.prev is LRU
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        prev_node, next_node = node.prev, node.next
        assert prev_node is not None and next_node is not None
        prev_node.next = next_node
        next_node.prev = prev_node

    def _insert_front(self, node: Node) -> None:
        first = self.head.next
        assert first is not None
        node.prev = self.head
        node.next = first
        self.head.next = node
        first.prev = node

    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._insert_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])

        node = Node(key, value)
        self.cache[key] = node
        self._insert_front(node)

        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            assert lru is not None and lru is not self.head
            self._remove(lru)
            del self.cache[lru.key]
```

## Takeaway

Keeping `_remove` and `_insert_front` as two tiny, single-responsibility helpers turns a notoriously bug-prone pointer problem into 35 lines of code you can explain effortlessly on a whiteboard.
