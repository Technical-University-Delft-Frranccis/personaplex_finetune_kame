"""Check 3b (kame_finetune env) - does the KAME-layout model compute the same function?

    cd personaplex_finetune_kame
    python -m tools.compare_layouts /workspace/ft_init /workspace/ref.pt

Loads the converted kame_finetune directory, turns it into a plain KAME LMModel (the
layout training operates on), feeds it the same codes / oracle / oracle table that
dump_pp_reference.py used, and compares logits.

Reading the result: a wiring or layout bug (wrong attention split, wrong oracle slot, wrong
delays) gives relative error of order 1 and top-1 agreement near chance. bf16 kernel
differences between two implementations give relative error around 1e-2 or below. The
thresholds below only separate those two regimes; they are not tight tolerances.
"""

import sys

import torch

from models import MoshiForFinetuning

ft_dir, ref_path = sys.argv[1:3]
ref = torch.load(ref_path)

ft = MoshiForFinetuning.from_pretrained(ft_dir, device="cpu", dtype=torch.bfloat16)
lm = ft.to_original_moshi_lm()          # strict load: also proves every key maps 1:1
del ft
lm = lm.to("cuda").eval()
lm.oracle_emb.weight.data.copy_(ref["emb"].to("cuda", torch.bfloat16))

with torch.no_grad():
    out = lm(ref["codes"].cuda(), oracle_tokens=ref["oracle"].cuda())


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    """a = PersonaPlex logits, b = KAME logits. Delay-invalid positions are NaN on both sides."""
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert torch.equal(torch.isfinite(a), torch.isfinite(b)), f"{name}: valid-position masks differ"
    m = torch.isfinite(a)
    diff = (a - b)[m].abs()
    rel = (diff.mean() / a[m].std()).item()
    valid_pos = m.all(-1)
    agree = (a.nan_to_num().argmax(-1) == b.nan_to_num().argmax(-1))[valid_pos].float().mean().item()
    print(f"{name:6s} max|diff|={diff.max():.4f}  mean|diff|={diff.mean():.5f}  "
          f"relative={rel:.4f}  top1-agreement={agree:.4f}")
    return rel < 0.1 and agree > 0.9


ok_text = compare("text", ref["text"], out.text_logits.float().cpu())
ok_audio = compare("audio", ref["audio"], out.logits.float().cpu())
sys.exit(0 if (ok_text and ok_audio) else "PARITY FAILED: the two layouts do not compute the same function")