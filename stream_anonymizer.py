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

Optional (skip_nonspeech=True): Vor dem Block laeuft die BEATs-VAD (identische Entscheidung wie in blur_audio).
Ist in keinem 0,5-s-Fenster Sprache, wird NBSS uebersprungen und der Block unveraendert ausgegeben (das tut
blur_audio fuer Nicht-Sprach-Fenster ohnehin); der Links-Kontext wird trotzdem fortgeschrieben.
Das senkt den mittleren Rechenaufwand, nicht den schlechtesten Block (Dauergespraech).

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


def vad_flags_like_blur_audio(y_ref, sr, vad_model, vad_sr, speech_index, ths, device, pad_quirk=True):
    """Fensterweise Sprachentscheidung EXAKT wie in anonymizer.blur_audio (Referenzkanal, 0,5-s-Fenster, Hop 0,44 s).

    pad_quirk=True bildet den Original-Fehler nach: blur_audio vergleicht die ANZAHL der Fenster mit 8000 und haengt
    deshalb jedem Fenster (8000 - Anzahl) Nullen an (BEATs sieht ~1 s, davon ~50 % Nullen). Muss mit blur_audio
    uebereinstimmen, sonst wuerde --skip-nonspeech Sprache durchlassen; test_new_features.py prueft das.
    """
    import torch
    import torchaudio
    from anonymizer import AudioChunks
    y_n = AudioChunks(n=round(0.5 * sr), hop=round(0.440 * sr)).chunks_with_hop(y_ref)
    x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=vad_sr)(torch.tensor(y_n, dtype=torch.float32))
    x = x / torch.max(torch.abs(x))
    target = int(0.5 * 16000)
    n_cmp = x.shape[0] if pad_quirk else x.shape[-1]
    if n_cmp < target:
        x = torch.nn.functional.pad(x, (0, target - n_cmp))
    with torch.no_grad():
        inp = x.clone().detach().to(device)
        mask = torch.zeros(inp.size(0), inp.size(1)).bool().to(device)
        logit = vad_model.extract_features(inp, padding_mask=mask)[0][:, speech_index]
    return (logit > ths).cpu().numpy()


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

    def advance(self, y_mc):
        """Block ohne NBSS-Lauf: nur den rohen Links-Kontext fortschreiben (bleibt im RAM)."""
        if self.ctx > 0:
            x = y_mc if self.prev is None else np.concatenate([self.prev, y_mc], axis=-1)
            self.prev = x[:, -self.ctx:].copy()


class StreamAnonymizer:
    def __init__(self, separator, blur_kwargs, blur_fn=None, block_s=2.0, ctx_s=2.0, n_chan=4,
                 skip_nonspeech=False):
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
        self.skip_nonspeech, self.n_skipped = skip_nonspeech, 0

    @property
    def block_seconds(self):
        return self.block / self.sr

    @property
    def pending_seconds(self):
        return self.buf.shape[1] / self.sr

    def _has_speech(self, blk):
        """Gleiche VAD-Entscheidung wie blur_audio (Referenzkanal 0, 0,5-s-Fenster); True, wenn irgendein Fenster Sprache hat."""
        kw = self.blur_kwargs
        if kw.get("vad_model") is None:
            return True
        flags = vad_flags_like_blur_audio(blk[0], self.sr, kw["vad_model"], kw["vad_sr"], kw["vad_speech_index"],
                                          kw["vad_speech_ths"], kw["device"])
        return bool(np.any(flags))

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
            if self.skip_nonspeech and not self._has_speech(blk):
                self.ctxsep.advance(blk)
                self.ctxsep.t_sep.append(0.0)
                self.n_skipped += 1
                y = blk[0].copy()
            else:
                y = self.blur_fn(blk, self.sr, source_sep=self.ctxsep, **self.blur_kwargs)
            self.t_block.append(time.perf_counter() - t0)
            y = np.asarray(y, dtype=np.float32)
            if y.shape[-1] != self.block:
                raise RuntimeError(f"Ausgabe {y.shape[-1]} != Block {self.block} Samples: blur_audio hat "
                                   f"vermutlich Rohsignal aufgefuellt -> Block wird nicht ausgegeben")
            outs.append(y)
        return outs