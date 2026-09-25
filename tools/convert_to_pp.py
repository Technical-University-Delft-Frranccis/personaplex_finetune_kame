import argparse
import re
import torch
from safetensors.torch import save_file
from models import MoshiForFinetuning

ATTN = re.compile(r"^(?P<prefix>.*\.self_attn\.)(?P<kind>in_projs|out_projs)\.(?P<i>\d+)\.weight$")

def kame_to_personaplex_layout(sd: dict) -> dict:
    """Inverse of KAME's _load_hook: refuse per-step attention linears to load with the personaplex layout."""
    out, parts = {}, {}
    for name, t in sd.items():
        m = ATTN.match(name)
        if m is None:
            out[name] = t
            continue
        parts.setdefault((m["prefix"], m["kind"]), {})[int(m["i"])] = t
    for (prefix, kind), by_step in parts.items():
        fused = torch.cat([by_step[i] for i in range(len(by_step))], dim=0)
        target = "in_proj_weight" if kind == "in_projs" else "out_proj.weight"
        out[prefix + target] = fused
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft_dir", required=True, help="kame_finetune checkpoint dir")
    p.add_argument("--out", required=True, help="e.g. personaplex_oracle_ft.safetensors")
    a = p.parse_args()

    ft = MoshiForFinetuning.from_pretrained(a.ft_dir, device="cpu", dtype=torch.bfloat16)
    lm = ft.to_original_moshi_lm()                     # undoes the ZeRO-3 gating renames
    sd = kame_to_personaplex_layout(lm.state_dict())
    save_file({k: v.contiguous() for k, v in sd.items()}, a.out)
    print(f"exported {len(sd)} tensors -> {a.out}")


if __name__ == "__main__":
    main()