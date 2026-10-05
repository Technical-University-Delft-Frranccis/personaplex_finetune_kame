"""kame_finetune fp32 checkpoint dir -> PersonaPlex-layout safetensors that personaplex-klm can load.

    uv run --no-sync -m tools.convert_to_pp --ft_dir output/pilot_A/step_150_fp32 --out /workspace/ckpt/A_s150.safetensors

Use this, not tools/clean_moshi.py: clean_moshi keeps KAME's split attention layout (in_projs.N / out_projs.N),
which personaplex-klm's loader rejects, and with --remove_modules_for_user_stream it cuts dep_q back to 8.

Change vs. the first version: the oracle embedding mode is resolved from the training config.json (the parent of
--ft_dir, like clean_moshi does) or --oracle_embedding_mode, and applied before export. In "tie" mode training reads
oracle tokens through text_emb, so the separate oracle_emb tensor in the checkpoint is stale until it is copied over;
exporting without that copy would ship an untrained oracle table.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from models import MoshiForFinetuning

ATTN = re.compile(r"^(?P<prefix>.*\.self_attn\.)(?P<kind>in_projs|out_projs)\.(?P<i>\d+)\.weight$")
MODES = ("separate", "tie")


def kame_to_personaplex_layout(sd: dict) -> dict:
    """Inverse of KAME's _load_hook: re-fuse per-step attention linears into the personaplex layout."""
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


def resolve_mode(ft_dir: str, cli_mode: str | None) -> tuple[str, str]:
    cfg = Path(ft_dir).parent / "config.json"
    meta = None
    if cfg.is_file():
        meta = json.loads(cfg.read_text(encoding="utf-8")).get("oracle_embedding_mode")
        if meta is not None and meta not in MODES:
            raise ValueError(f"{cfg} records invalid oracle_embedding_mode {meta!r}")
    if meta is not None and cli_mode is not None and meta != cli_mode:
        raise ValueError(f"oracle_embedding_mode mismatch: {cfg} says {meta!r}, CLI says {cli_mode!r}")
    if meta is not None:
        return meta, str(cfg)
    if cli_mode is not None:
        return cli_mode, "CLI (no oracle_embedding_mode in a training config.json)"
    raise SystemExit(f"cannot tell the oracle embedding mode: no oracle_embedding_mode in {cfg}. "
                     "Pass --oracle_embedding_mode separate|tie (the mode the run trained with).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ft_dir", required=True, help="kame_finetune fp32 checkpoint dir (model.safetensors + moshi_lm_kwargs.json)")
    p.add_argument("--out", required=True, help="e.g. personaplex_oracle_ft.safetensors")
    p.add_argument("--oracle_embedding_mode", choices=MODES, default=None)
    p.add_argument("--pad_id", type=int, default=3, help="text PAD id, only used for the printed check")
    a = p.parse_args()

    mode, source = resolve_mode(a.ft_dir, a.oracle_embedding_mode)
    ft = MoshiForFinetuning.from_pretrained(a.ft_dir, device="cpu", dtype=torch.bfloat16)
    ft.materialize_oracle_embedding_for_checkpoint_(mode)
    lm = ft.to_original_moshi_lm()                     # undoes the ZeRO-3 gating renames, strict load
    sd = kame_to_personaplex_layout(lm.state_dict())

    o, t = sd["oracle_emb.weight"], sd["text_emb.weight"]
    print(f"oracle_embedding_mode={mode} (from {source}) | oracle_emb max|w| {o.float().abs().max().item():.4g} | "
          f"PAD row norm {o[a.pad_id].float().norm().item():.4g} | identical to text_emb: {torch.equal(o, t)}")
    save_file({k: v.contiguous() for k, v in sd.items()}, a.out)
    print(f"exported {len(sd)} tensors -> {a.out}")


if __name__ == "__main__":
    main()