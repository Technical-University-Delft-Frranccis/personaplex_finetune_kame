"""Which parameter groups did training actually change? (finetune env; safetensors + torch only)

    uv run --no-sync python -m tools.check_trained_params \
        --init /workspace/ft_init/model.safetensors --trained output/smoke/step_5_fp32/model.safetensors

Run it after the smoke run. It answers two questions finetune.py's flags don't show directly:
  * is oracle_emb in the trainable set under --parameters_to_finetune tempformer?  (it must change)
  * is the depth transformer really frozen?                                      (it must not change)
Both files must be in the MoshiForFinetuning layout (ft_init and zero_to_fp32 output are). Tensors are compared
one at a time, so memory stays low.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from safetensors import safe_open

# first matching prefix wins
GROUPS = [
    ("oracle_emb.", "oracle_emb (oracle table)"),
    ("text_emb.", "text_emb"),
    ("text_linear.", "text_linear (text head)"),
    ("transformer.", "temporal transformer"),
    ("out_norm", "out_norm"),
    ("depformer", "depth transformer (depformer*)"),
    ("linears.", "depth heads (linears)"),
    ("emb.", "audio input embeddings (emb)"),
]


def group_of(key: str) -> str:
    for prefix, name in GROUPS:
        if key.startswith(prefix):
            return name
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--pad_id", type=int, default=3)
    args = ap.parse_args()

    stats = defaultdict(lambda: {"tensors": 0, "changed": 0, "max_diff": 0.0})
    with safe_open(args.init, "pt") as a, safe_open(args.trained, "pt") as b:
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb:
            print(f"KEY MISMATCH: only in init {sorted(ka - kb)[:5]}  only in trained {sorted(kb - ka)[:5]}")
        for k in sorted(ka & kb):
            x, y = a.get_tensor(k).float(), b.get_tensor(k).float()
            if x.shape != y.shape:
                print(f"SHAPE MISMATCH {k}: {tuple(x.shape)} vs {tuple(y.shape)}")
                continue
            d = (x - y).abs().max().item() if x.numel() else 0.0
            s = stats[group_of(k)]
            s["tensors"] += 1
            s["changed"] += d > 0
            s["max_diff"] = max(s["max_diff"], d)
            if k == "oracle_emb.weight":
                rows = int(((x - y).abs().sum(dim=1) > 0).sum())
                print(f"oracle_emb: {rows}/{x.shape[0]} rows changed | PAD row norm init "
                      f"{x[args.pad_id].norm().item():.4g} -> trained {y[args.pad_id].norm().item():.4g}")

    print(f"\n{'group':34s} {'tensors':>7s} {'changed':>7s} {'max |diff|':>11s}")
    for name, s in sorted(stats.items()):
        print(f"{name:34s} {s['tensors']:7d} {s['changed']:7d} {s['max_diff']:11.3g}")
    o = stats.get("oracle_emb (oracle table)", {"changed": 0})
    d = stats.get("depth transformer (depformer*)", {"changed": 0})
    print("\nverdict:",
          "oracle_emb TRAINS" if o["changed"] else "oracle_emb did NOT change -> it is not in the trainable set",
          "|", "depformer frozen" if not d["changed"] else "depformer CHANGED -> not frozen")


if __name__ == "__main__":
    main()
