"""Word-level forced alignment of KNOWN text against a single-speaker clip.

Uses torchaudio's MMS_FA bundle (available in the torchaudio 2.4.1 pinned by kame_finetune).
Because we align the script text we already have against the isolated TTS clip, this is
much more accurate than running ASR on the mixed stereo dialogue.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass
class AlignedWord:
    word: str       # display form (original casing/punctuation), goes into text/*.json
    start: float    # seconds, relative to the clip
    end: float
    score: float    # mean CTC probability of the word's tokens, 0..1


class WordAligner:
    def __init__(self, device: str = "cuda"):
        self.bundle = torchaudio.pipelines.MMS_FA
        self.device = device
        self.model = self.bundle.get_model(with_star=False).to(device).eval()
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        # Characters the acoustic model knows; '-' is the CTC blank, '*' the star token.
        self.charset = {c for c in self.bundle.get_dict() if c not in ("-", "*")}
        self.sample_rate = int(self.bundle.sample_rate)

    def normalize(self, word: str) -> str:
        w = word.lower().replace("\u2019", "'")
        return "".join(c for c in w if c in self.charset)

    @torch.inference_mode()
    def align(self, wav: torch.Tensor, sr: int, text: str) -> list[AlignedWord]:
        """wav: [1, T] or [T] float tensor. Returns one entry per alignable display word.

        Display words that normalize to nothing (e.g. '...', '-') are dropped: they carry no
        speech and would otherwise produce zero-length tokens in the text stream.
        """
        if wav.dim() == 1:
            wav = wav[None]
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        wav16 = torchaudio.functional.resample(wav, sr, self.sample_rate)

        display = text.split()
        pairs = [(d, self.normalize(d)) for d in display]
        pairs = [(d, n) for d, n in pairs if n]
        if not pairs:
            return []

        emission, _ = self.model(wav16.to(self.device))
        n_tokens = sum(len(n) for _, n in pairs)
        if emission.size(1) < n_tokens:
            raise ValueError(f"clip too short for text ({emission.size(1)} frames < {n_tokens} chars)")
        spans = self.aligner(emission[0], self.tokenizer([n for _, n in pairs]))

        sec_per_frame = wav16.size(1) / emission.size(1) / self.sample_rate
        out: list[AlignedWord] = []
        for (disp, _), word_spans in zip(pairs, spans, strict=True):
            start = word_spans[0].start * sec_per_frame
            end = word_spans[-1].end * sec_per_frame
            score = float(sum(s.score for s in word_spans) / len(word_spans))
            out.append(AlignedWord(disp, round(start, 3), round(end, 3), score))
        return out
