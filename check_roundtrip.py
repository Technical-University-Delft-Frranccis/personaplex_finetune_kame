"""Check 0 - are two safetensors files bit-identical (same keys, dtypes, bytes)?

    python check_roundtrip.py /workspace/checkpoints/personaplex_oracle.safetensors \
                              /workspace/checkpoints/roundtrip.safetensors

Passing means the PersonaPlex -> KAME layout -> PersonaPlex conversion lost nothing.
It does NOT prove the KAME-layout model computes the same function (see compare_layouts.py).
"""

import sys

import torch
from safetensors import safe_open

a_path, b_path = sys.argv[1:3]
with safe_open(a_path, "pt") as fa, safe_open(b_path, "pt") as fb:
    ka, kb = set(fa.keys()), set(fb.keys())
    if ka != kb:
        sys.exit(f"KEY MISMATCH\n  only in A: {sorted(ka - kb)[:8]}\n  only in B: {sorted(kb - ka)[:8]}")
    bad = []
    for k in sorted(ka):
        ta, tb = fa.get_tensor(k), fb.get_tensor(k)
        if ta.dtype != tb.dtype or ta.shape != tb.shape or not torch.equal(ta, tb):
            bad.append(k)
if bad:
    sys.exit(f"{len(bad)} tensors differ, first: {bad[:8]}")
print(f"bit-exact: {len(ka)} tensors identical")