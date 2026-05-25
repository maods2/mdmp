"""One-off: compare build_parameter_names vs build_design_matrix parent order."""
import sys

import numpy as np

from mdmp.utils import build_design_matrix, build_parameter_names

def mismatches(adj: np.ndarray, names: list[str]) -> list:
    n = adj.shape[0]
    T = 5
    dummy = np.zeros((T, n))
    out = []
    for c in range(n):
        _, pl = build_design_matrix(dummy, adj, c)
        pn = build_parameter_names(c, adj, names)
        conn_parents = []
        for name in pn:
            if "->" in name:
                conn_parents.append(names.index(name.split("->", 1)[0]))
        if conn_parents != pl:
            out.append((c, pl, conn_parents, pn))
    return out


def main() -> int:
    names = ["A", "B", "C"]
    adj = np.zeros((3, 3), dtype=int)
    adj[1, 1] = 1
    adj[0, 2] = 1
    m = mismatches(adj, names)
    sys.stdout.write(f"self-loop case mismatches: {len(m)}\n")
    for x in m:
        sys.stdout.write(f"  {x}\n")

    rng = np.random.default_rng(0)
    for trial in range(200):
        n = 6
        adj = np.triu(rng.integers(0, 2, (n, n)), 1)
        ns = [f"V{i}" for i in range(n)]
        if mismatches(adj, ns):
            sys.stdout.write(f"trial {trial} mismatch\n")
            return 1
    sys.stdout.write("200 random DAGs: no mismatch\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
