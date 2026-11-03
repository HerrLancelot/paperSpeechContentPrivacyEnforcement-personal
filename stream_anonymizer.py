"""Weg A: blockweise Anonymisierung mit Ringpuffer (4 Kanaele, 16 kHz) und Links-Kontext.

Ablauf pro Block (alle ~2,3 s Audio):
    Ringpuffer --Block--> blur_audio(block, source_sep=ContextSeparator)
        ContextSeparator haengt den rohen Links-Kontext (ctx_s) vor den Block, laesst NBSS auf
        [Kontext + Block] laufen und gibt nur den Block-Teil zurueck (Modell ist kausal).
        blur_audio selbst (Reversal-Segmente, BEATs, Splicing) laeuft unveraendert nur auf dem Block.

Rohaudio: nur dieser Puffer und der Kontext im ContextSeparator halten Rohdaten. Nach aussen geht
ausschliesslich der Rueckgabewert von blur_audio. Die Blocklaenge liegt auf dem BEATs-Raster
(n + k*hop), damit blur_audio keinen unbedeckten Rest am Blockende mit Rohsignal auffuellt
(sonst wuerden pro Block ~0,18 s Original ausgegeben). Das wird pro Block geprueft.

Nur numpy; blur_audio wird uebergeben oder lazy aus anonymizer.py importiert.
"""
import time

import numpy as np

BEATS_N_S, BEATS_HOP_S = 0.500, 0.440   # wie AudioChunks(n=0.5 s, hop=0.44 s) in blur_audio


def snap_block(block_s, sr):
    """Kleinste Blocklaenge >= block_s auf dem BEATs-Raster n + k*hop (k >= 2)."""
    n, hop = round(BEATS_N_S * sr), round(BEATS_HOP_S * sr)
    k = max(2, int(np.ceil((block_s * sr - n) / hop)))
    return n + k * hop


class ContextSeparator:
    """source_sep-Adapter fuer blur_audio: .separate([C,T], sr) -> (voice[T], other[T])."""

    def __init__(self, sep, ctx_s=2.0):
        self.sep, self.sr = sep, sep.sr
        self.ctx = int(round(ctx_s * self.sr))
        self.prev = None
        self.t_sep = []          # Sekunden je Aufruf

    def separate(self, y_mc, sr):
        assert sr == self.sr, (sr, self.sr)
        L = y_mc.shape[-1]
        x = y_mc if self.prev is None else np.concatenate([self.prev, y_mc], axis=-1)
        t0 = time.perf_counter()
        voice, other = self.sep.separate(x)       # [ctx+L]; .cpu().numpy() synchronisiert die GPU
        self.t_sep.append(time.perf_counter() - t0)
        self.prev = x[:, -self.ctx:].copy() if self.ctx > 0 else None
        return voice[-L:], other[-L:]


class StreamAnonymizer:
    def __init__(self, separator, blur_kwargs, blur_fn=None, block_s=2.0, ctx_s=2.0, n_chan=4):
        """
        separator:   NBSSSeparator (am besten seg=None)
        blur_kwargs: Argumente fuer blur_audio ohne y, sr, source_sep (segmin, segmax, framemin, ...,
                     device, vad_model, vad_sr, vad_speech_index, vad_speech_ths, method)
        """
        self.sr, self.n_chan = separator.sr, n_chan
        self.block = snap_block(block_s, self.sr)
        self.ctxsep = ContextSeparator(separator, ctx_s)
        if blur_fn is None:
            from anonymizer import blur_audio as blur_fn
        self.blur_fn, self.blur_kwargs = blur_fn, dict(blur_kwargs)
        self.buf = np.zeros((n_chan, 0), dtype=np.float32)
        self.t_block = []        # Sekunden je Block (gesamt, ohne Warten auf Eingang)

    @property
    def block_seconds(self):
        return self.block / self.sr

    @property
    def pending_seconds(self):
        return self.buf.shape[1] / self.sr

    def push(self, x):
        """x: [C, n] float (beliebige n, z. B. Callback-Chunks) -> Liste anonymisierter Bloecke [block]."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != self.n_chan:
            raise ValueError(f"erwartet [{self.n_chan}, n], erhalten {x.shape}")
        self.buf = np.concatenate([self.buf, x], axis=1)
        outs = []
        while self.buf.shape[1] >= self.block:
            blk = np.ascontiguousarray(self.buf[:, :self.block])
            self.buf = self.buf[:, self.block:].copy()
            t0 = time.perf_counter()
            y = self.blur_fn(blk, self.sr, source_sep=self.ctxsep, **self.blur_kwargs)
            self.t_block.append(time.perf_counter() - t0)
            y = np.asarray(y, dtype=np.float32)
            if y.shape[-1] != self.block:
                raise RuntimeError(f"Ausgabe {y.shape[-1]} != Block {self.block} Samples: blur_audio hat "
                                   f"vermutlich Rohsignal aufgefuellt -> Block wird nicht ausgegeben")
            outs.append(y)
        return outs