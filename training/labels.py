"""Stage / progress labels from grind count."""

from __future__ import annotations


def cls3(grind: int) -> int:
    if grind <= 40:
        return 0
    if grind <= 110:
        return 1
    return 2


def cls5(grind: int) -> int:
    for i, e in enumerate([34, 68, 102, 136, 169]):
        if grind <= e:
            return i
    return 4


def progress(grind: int) -> float:
    return float(grind) / 169.0


def cls3_edges(grind: int, e1: int = 40, e2: int = 110) -> int:
    if grind <= e1:
        return 0
    if grind <= e2:
        return 1
    return 2
