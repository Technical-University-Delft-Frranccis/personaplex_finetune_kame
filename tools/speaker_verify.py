"""Speaker-verification embeddings (WavLM x-vector) for identity checks on TTS output.

PersonaPlex reports speaker similarity with a WavLM-TDNN verifier; microsoft/wavlm-base-plus-sv
is the closest openly available equivalent. Used by build_voice_bank.py (select emotional
references that keep the hero identity) and synthesize_turns.py (reject drifted turns).
Runs in the TTS environment (qwen-tts already installs transformers).
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


class SpeakerVerifier:
    def __init__(self, device: str = "cuda", model_id: str = "microsoft/wavlm-base-plus-sv"):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def embed(self, wav: np.ndarray | torch.Tensor, sr: int) -> torch.Tensor:
        x = torch.as_tensor(wav, dtype=torch.float32)
        if x.dim() == 2:
            x = x.mean(0)
        x16 = torchaudio.functional.resample(x, sr, 16000)
        inputs = self.fe(x16.numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.model(**inputs).embeddings, dim=-1)[0].cpu()

    def embed_file(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path, backend="soundfile")
        return self.embed(wav, sr)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(a @ b)


def medoid_index(embeddings: list[torch.Tensor]) -> int:
    """Index of the embedding with the highest mean similarity to all others: the most
    typical realisation of a voice description (used to pick a hero voice's anchor)."""
    m = torch.stack(embeddings)
    sims = m @ m.T
    return int(sims.mean(dim=1).argmax())