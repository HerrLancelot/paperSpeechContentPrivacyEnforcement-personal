"""
Blockweiser Live-Anonymisierer mit Mehrkanal-Ausgabe (Gegenstueck zu stream_anonymizer.StreamAnonymizer).

Schnittstelle wie dort, damit live_anonymize.py / measure_rtf_blockwise.py nur die Klasse tauschen:
  push(chunk[C, n]) -> Liste anonymisierter Bloecke, je [C, block]
  .block, .block_seconds, .pending_seconds, .t_block (Gesamtzeit je Block), .ctxsep.t_sep (Zeit im Separator je Block)

Je Block (36160 Samples = 5 BEATs-Fenster, 2,26 s bei 16 kHz; so bleibt kein ungeschuetzter Rest):
  1. Rohblock + bis zu ctx_s Sekunden rohen Links-Kontext (nur im RAM) -> Separator, 4 gerollte Laeufe
     (nacheinander oder als Batch), vom Ergebnis zaehlt nur der Blockanteil.
  2. mc_anonymizer.blur_audio_mc: gemeinsamer Plan (Segmente, Reversal) und gemeinsame VAD fuer alle Kanaele.
Rohaudio wird nie geschrieben; zurueckgegeben wird nur das anonymisierte Ergebnis.

Optional (skip_nonspeech=True): VAD vorab (gleiche Entscheidung wie in blur_audio_mc, 'ref' = Kanal 0, 'any' = ODER);
ohne Sprache in allen Fenstern entfallen NBSS (4 Rollungen) und Splicing, der Block geht unveraendert durch.
Der rohe Links-Kontext wird trotzdem fortgeschrieben. Senkt den Mittelwert, nicht den schlechtesten Block.
"""
import math
import time
import types

import numpy as np

import mc_anonymizer
from stream_anonymizer import snap_block   # gleiches Blockraster wie StreamAnonymizer

class StreamAnonymizerMC:
    def __init__(self, sep, kw, block_s=2.0, ctx_s=2.0, batch_roll=False, vad_mode="ref",
                 tail_mode="last_window", seed=None, skip_nonspeech=False):
        self.sep, self.kw = sep, dict(kw)
        self.sr, self.C = sep.sr, sep.N_CHN
        self.block = snap_block(block_s, self.sr)                   # 5 Fenster = 36160 Samples bei 16 kHz
        self.block_seconds = self.block / self.sr
        self.ctx = int(ctx_s * self.sr)
        self.batch_roll = batch_roll and hasattr(sep, "separate_batch")
        self.vad_mode, self.tail_mode, self.seed = vad_mode, tail_mode, seed
        self.buf = np.zeros((self.C, 0), np.float32)
        self.hist = np.zeros((self.C, 0), np.float32)               # roher Links-Kontext, nur im Speicher
        self.t_block = []
        self.ctxsep = types.SimpleNamespace(t_sep=[])
        self.n_blocks = 0
        self.skip_nonspeech, self.n_skipped = skip_nonspeech, 0

    @property
    def pending_seconds(self):
        return self.buf.shape[1] / self.sr

    def _sync(self):
        dev = self.kw.get("device")
        if getattr(dev, "type", dev) == "cuda":
            import torch
            torch.cuda.synchronize()

    def _separate(self, x):
        """x: [C, ctx+block] -> voice, other [C, ctx+block]; Zeile c = Schaetzung am Originalkanal c."""
        C = x.shape[0]
        rolls = [np.roll(x, -c, axis=0) for c in range(C)]
        if self.batch_roll:
            v, o = self.sep.separate_batch(np.stack(rolls))
            return np.asarray(v, np.float32), np.asarray(o, np.float32)
        v = np.zeros_like(x)
        o = np.zeros_like(x)
        for c in range(C):
            vc, oc = self.sep.separate(rolls[c])
            v[c, :len(vc)], o[c, :len(oc)] = vc[:x.shape[1]], oc[:x.shape[1]]
        return v, o

    def _has_speech(self, blk):
        """Gleiche VAD-Entscheidung wie blur_audio_mc; True, wenn irgendein 0,5-s-Fenster Sprache hat."""
        kw = self.kw
        if kw.get("vad_model") is None:
            return True
        for c in (range(self.C) if self.vad_mode == "any" else [0]):
            flags = mc_anonymizer.vad_speech_flags(blk[c], self.sr, kw["vad_model"], kw["vad_sr"],
                                                   kw["vad_speech_index"], kw["vad_speech_ths"], kw["device"])
            if np.any(flags):
                return True
        return False

    def _process(self, blk):
        t_start = time.perf_counter()
        if self.skip_nonspeech and not self._has_speech(blk):
            out = blk.copy()
            self._sync()
            self.t_block.append(time.perf_counter() - t_start)
            self.ctxsep.t_sep.append(0.0)
            self.n_blocks += 1
            self.n_skipped += 1
            self.hist = np.concatenate([self.hist, blk], axis=1)[:, -max(self.ctx, 1):]
            return out
        ctxblk = np.concatenate([self.hist[:, -self.ctx:], blk], axis=1) if self.ctx else blk
        self._sync()
        t0 = time.perf_counter()
        v, o = self._separate(ctxblk)
        self._sync()
        t_sep = time.perf_counter() - t0
        out, _ = mc_anonymizer.blur_audio_mc(
            blk, self.sr, voice=v[:, -self.block:], other=o[:, -self.block:],
            vad_mode=self.vad_mode, tail_mode=self.tail_mode,
            seed=None if self.seed is None else self.seed + self.n_blocks, **self.kw)
        self._sync()
        if out.shape != blk.shape:
            raise RuntimeError(f"Ausgabelaenge {out.shape} != Block {blk.shape}")
        self.t_block.append(time.perf_counter() - t_start)
        self.ctxsep.t_sep.append(t_sep)
        self.n_blocks += 1
        self.hist = np.concatenate([self.hist, blk], axis=1)[:, -max(self.ctx, 1):]
        return out

    def push(self, chunk):
        assert chunk.shape[0] == self.C, chunk.shape
        self.buf = np.concatenate([self.buf, chunk.astype(np.float32, copy=False)], axis=1)
        res = []
        while self.buf.shape[1] >= self.block:
            blk, self.buf = self.buf[:, :self.block], self.buf[:, self.block:]
            res.append(self._process(blk))
        return res