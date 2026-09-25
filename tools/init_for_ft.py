import argparse
from copy import deepcopy
import torch
from safetensors.torch import load_file
from kame.models import LMModel, loaders
from models import MoshiForFinetuning
from models.oracle_embedding_utils import validate_oracle_embedding_checkpoint_load

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src",required=True,help="personaplex_oracle.safetensors")
    p.add_argument("--save_dir",required=True)
    p.add_argument("--dtype",default = "bfloat16")
    a = p.parse_args()
    dtype = getattr(torch,a.dtype)

    #adapts kame's finetuning which only uses dep_q = 8 (moshi base)
    # hence we need to update the kwargs
    kwargs = deepcopy(loaders._lmkwargs)
    kwargs.update(dep_q=16,depformer_contex=16)
    
    lm = LMModel(device="cpu",dtype=dtype,**kwargs)
    sd = load_file(a.src,device="cpu")
    result = lm.load_state_dict(sd,strict=False) #split fused attention weights from personaplex
    oracle_missing = validate_oracle_embedding_checkpoint_load(
        lm,missing_keys = result.missing_keys, unexpected=result.unexpected_keys,
        context=f"loading{a.src}",
    )
    assert not oracle_missing, "checkpoint must already contain oracle_emb"
    assert torch.equal(lm.oracle_emb.weight,sd["oracle_emb.weight"].to(dtype))

    ft = MoshiForFinetuning.from_original_moshi_lm(moshi_lm = lm, mosi_lm_kwargs = kwargs)
    ft.save_pretrained(a.save_dir) #json of safetensors + kwargs
    print(f"kame_compatible_finetune_model --> {a.save_dir}")

if __name__ = "__main__":
    main()