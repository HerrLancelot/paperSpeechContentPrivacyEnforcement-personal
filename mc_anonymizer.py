"""
Multichannel version of the Tailleur et al. anonymization (segment-wise waveform reversal).

Idee
----
Das Mono-Original (`blur_audio`) trifft pro Aufruf zufaellige Entscheidungen
(Segmentgrenzen, Splicing-Indizes, Reversal-Flags) und eine VAD-Entscheidung.
Fuer Mehrkanal muessen diese Entscheidungen EINMAL getroffen und dann auf alle
Kanaele identisch angewendet werden, sonst gehen die Zwischenkanal-Beziehungen
(Zeitversatz, Pegel, Phase) der erhaltenen Anteile kaputt.

Ablauf (`blur_audio_mc`):
  1. Sprachschaetzung je Kanal: Kanalachse so rollen, dass Kanal c an Position 0
     liegt, Separator laufen lassen (Output 0 = Sprache, 1 = Background).
  2. Plan aus der Referenzsprache (Kanal `ref`): aeussere Segmente
     (RandomAudioChunks), je Segment Silence-Splitting, optional Shuffle,
     Reversal-Flags. Reihenfolge der RNG-Aufrufe wie im Original.
  3. Plan auf alle Kanaele der Sprachschaetzung anwenden (gleiche Indizes).
  4. VAD (BEATs, 0,5-s-Fenster) einmal berechnen ('ref' = nur Referenzkanal,
     'any' = ODER ueber alle Kanaele) und auf alle Kanaele anwenden.
  5. Je Fenster: Sprache -> reversed voice + Background-Schaetzung, sonst Original.

Mit gleichem Seed ist Kanal `ref` der Ausgabe identisch zur Mono-Pipeline
(siehe test_mc_anonymizer.py).

Hinweis: `random` und `np.random` werden global geseedet, weil
RandomAudioChunks und random_splicing diese globalen Generatoren nutzen.
"""
import random

import numpy as np

from anonymizer import RandomAudioChunks, AudioChunks
from utils.random_splicing import split_wav_naive, apply_crossfade


# ----------------------------------------------------------------------------
# 1. Sprachschaetzung je Kanal (gerollte Referenz)
# ----------------------------------------------------------------------------
def separate_mc_rolled(sep_ref0, mix, sep_batch=None):
    """
    mix: [C, T]. sep_ref0(x[C, T]) -> (voice[T], other[T]), geschaetzt am
    Referenzkanal 0 von x (so arbeitet der NBSS-Separator intern).
    Rueckgabe: voice[C, T], other[C, T], wobei Zeile c die Schaetzung am
    Originalkanal c ist.
    Annahme: Arraygeometrie ist ein symmetrischer Ring (z.B. Quadrat), sodass
    ein zyklisches Rollen einer Drehung des Arrays entspricht.
    sep_batch (optional): f(xs[B, C, T]) -> (speech[B, T], background[B, T]). Wenn gesetzt,
    laufen alle C Rollungen in einem Batch (mehr Speicher, bessere GPU-Auslastung);
    sonst nacheinander ueber sep_ref0.
    """
    C, T = mix.shape
    if sep_batch is not None:
        v, o = sep_batch(np.stack([np.roll(mix, -c, axis=0) for c in range(C)]))
        voice = np.zeros((C, T), dtype=np.float32)
        other = np.zeros((C, T), dtype=np.float32)
        n = min(T, v.shape[-1])
        voice[:, :n], other[:, :n] = v[:, :n], o[:, :n]
        return voice, other
    voice = np.zeros((C, T), dtype=np.float32)
    other = np.zeros((C, T), dtype=np.float32)
    for c in range(C):
        v, o = sep_ref0(np.roll(mix, -c, axis=0))
        voice[c, : len(v)] = v[:T]
        other[c, : len(o)] = o[:T]
    return voice, other


# ----------------------------------------------------------------------------
# 2. Plan (alle Zufallsentscheidungen, aus der Referenz abgeleitet)
# ----------------------------------------------------------------------------
def _splice_plan(seg_ref, sr, p_reverse, overlap, frame_length, mixframe, top_db):
    """Entspricht utils.random_splicing.random_splicing, aber nur die Entscheidungen."""
    frame_length = frame_length or 0
    hop_length = frame_length if frame_length != 0 else int(sr * 512 / 44100)
    frame_len = 4 * frame_length if frame_length != 0 else int(sr * 2048 / 44100)
    overlap_length = int(frame_len * overlap)

    n = len(seg_ref)
    peak = np.max(np.abs(seg_ref)) if n else 0.0
    indices = None
    if peak > 0:
        try:
            indices = split_wav_naive(
                seg_ref / peak, sr, top_db=top_db, frame_length=frame_len, hop_length=hop_length
            )
        except IndexError:  # librosa.split ohne nicht-stille Intervalle
            indices = None
    if indices is None:  # Segment still: unveraendert lassen (Original wuerde NaN/IndexError liefern)
        return np.array([[0, n]]), [False], overlap_length

    if mixframe:
        np.random.shuffle(indices)
    flags = [bool(np.random.rand(1)[0] <= p_reverse) for _ in range(len(indices))]
    return indices, flags, overlap_length


def make_plan(
    y_ref_voice, sr, segmin, segmax, seghopmin_perc, seghopmax_perc,
    p_reverse, framehopmin_perc, mixframe, framemin, top_db=6,
):
    """Plan fuer ein Signal der Laenge T; gibt (chunker, plan) zurueck."""
    chunker = RandomAudioChunks(
        segmin=segmin, segmax=segmax, seghopmin_perc=seghopmin_perc, seghopmax_perc=seghopmax_perc
    )
    # Index-Trick: chunks_with_hop auf arange liefert die Sample-Indizes je Segment
    # (inkl. Ueberlappung und Spiegelung an den Raendern), gueltig fuer jeden Kanal.
    idx_chunks = chunker.chunks_with_hop(np.arange(len(y_ref_voice)), mixframe=False)

    plan = []
    for idx in idx_chunks:
        seg_ref = np.array(y_ref_voice[idx], dtype=np.float64)
        ind, flags, ov = _splice_plan(
            seg_ref, sr, p_reverse, overlap=(1 - framehopmin_perc),
            frame_length=framemin, mixframe=mixframe, top_db=top_db,
        )
        plan.append((idx, ind, flags, ov))
    return chunker, plan


def apply_plan(y, chunker, plan):
    """Plan auf ein Kanalsignal y[T] anwenden."""
    segs = []
    for idx, ind, flags, ov in plan:
        seg = np.array(y[idx], dtype=np.float64)
        remix = []
        for (start, end), rev in zip(ind, flags):
            w = seg[start:end]
            if rev:
                w = w[::-1]
            if len(remix) == 0:
                remix.append(w)
            elif len(remix[-1]) > 0:
                remix[-1] = apply_crossfade(remix[-1], w, ov)
            else:
                remix.append(w)
        segs.append(np.hstack(remix))
    return chunker.concat_with_hop(segs)


# ----------------------------------------------------------------------------
# 3. VAD (BEATs), einmal fuer alle Kanaele
# ----------------------------------------------------------------------------
def vad_speech_flags(y_ref, sr, vad_model, vad_sr, speech_index, ths, device):
    """Fensterweise Sprachentscheidung wie in blur_audio; Rueckgabe bool[n_windows]."""
    chunker = AudioChunks(n=round(0.5 * sr), hop=round(0.440 * sr))
    y_n = chunker.chunks_with_hop(y_ref)
    if vad_model is None:
        return np.ones(len(y_n), dtype=bool)

    import torch
    import torchaudio

    x = torch.tensor(y_n, dtype=torch.float32)
    if sr != vad_sr:
        x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=vad_sr)(x)
    x = x / torch.max(torch.abs(x))
    target = int(0.5 * vad_sr)
    if x.shape[-1] < target:
        x = torch.nn.functional.pad(x, (0, target - x.shape[-1]))
    with torch.no_grad():
        x = x.to(device)
        mask = torch.zeros(x.size(0), x.size(1)).bool().to(device)
        logit = vad_model.extract_features(x, padding_mask=mask)[0][:, speech_index]
    return (logit > ths).cpu().numpy()


# ----------------------------------------------------------------------------
# 4. Gesamtpipeline
# ----------------------------------------------------------------------------
def blur_audio_mc(
    y, sr, segmin, segmax, framemin, sep_ref0=None,
    seghopmin_perc=0.95, seghopmax_perc=0.95, framehopmin_perc=0.80,
    mixframe=True, reverse="backward", ref=0,
    vad_model=None, vad_sr=16000, vad_speech_index=None, vad_speech_ths=0.3,
    vad_mode="ref", device="cpu", seed=None, method="random_splicing_reverse",
    tail_mode="last_window", sep_batch=None, voice=None, other=None,
):
    """
    y: [C, T] Mehrkanal-Mischung. Rueckgabe: (y_out[C, T], info).
    vad_mode: 'ref' (nur Kanal ref, vergleichbar mit Mono) oder 'any' (ODER ueber Kanaele).
    voice, other: optional bereits geschaetzte Sprache/Hintergrund [C, T] (z. B. aus dem Streaming mit
      Links-Kontext); dann wird nicht getrennt und sep_ref0 darf None sein.
    tail_mode: Rest hinter dem letzten vollstaendigen 0,5-s-Fenster (bei 10 s: ca. 0,26 s).
      'raw'         = wie blur_audio im Original: Rest unveraendert aus y (Sprache bleibt dort hoerbar!)
      'last_window' = Rest bekommt eine eigene VAD-Entscheidung (letztes, am Ende ausgerichtetes
                      0,5-s-Fenster) und wird bei Sprache ebenfalls anonymisiert.
    """
    if method != "random_splicing_reverse":
        raise NotImplementedError("Nur random_splicing_reverse hat einen teilbaren Plan.")
    y = np.asarray(y, dtype=np.float32)
    C, T = y.shape

    if voice is None:
        voice, other = separate_mc_rolled(sep_ref0, y, sep_batch=sep_batch)
    else:
        voice = np.asarray(voice, dtype=np.float32)
        other = np.asarray(other, dtype=np.float32)
        assert voice.shape == y.shape and other.shape == y.shape, (voice.shape, other.shape, y.shape)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    p_reverse = 1.0 if reverse == "backward" else 0.0 if reverse == "forward" else 0.5
    chunker, plan = make_plan(
        voice[ref], sr, segmin, segmax, seghopmin_perc, seghopmax_perc,
        p_reverse, framehopmin_perc, mixframe, framemin,
    )

    r_voice = np.zeros((C, T), dtype=np.float64)
    for c in range(C):
        r = apply_plan(voice[c], chunker, plan)
        r_voice[c, : min(T, len(r))] = r[:T]

    # VAD
    if vad_mode == "any":
        flags = np.zeros(len(AudioChunks(round(0.5 * sr), round(0.44 * sr)).chunks_with_hop(y[0])), bool)
        for c in range(C):
            flags |= vad_speech_flags(y[c], sr, vad_model, vad_sr, vad_speech_index, vad_speech_ths, device)
    else:
        flags = vad_speech_flags(y[ref], sr, vad_model, vad_sr, vad_speech_index, vad_speech_ths, device)

    # VAD-Entscheidung fuer den Rest hinter dem letzten Fenster
    n_win = round(0.5 * sr)
    tail_speech = False
    if tail_mode == "last_window" and vad_model is not None and T >= n_win:
        chans = range(C) if vad_mode == "any" else [ref]
        tail_speech = any(
            bool(vad_speech_flags(y[c][T - n_win:], sr, vad_model, vad_sr, vad_speech_index,
                                  vad_speech_ths, device)[0])
            for c in chans
        )
    elif tail_mode == "last_window" and vad_model is None:
        tail_speech = True  # ohne VAD wird alles anonymisiert
    elif tail_mode != "raw" and tail_mode != "last_window":
        raise ValueError(tail_mode)

    out = np.zeros((C, T), dtype=np.float32)
    thr = AudioChunks(n=round(0.5 * sr), hop=round(0.440 * sr))
    for c in range(C):
        y_n = thr.chunks_with_hop(y[c])
        rv_n = thr.chunks_with_hop(r_voice[c])
        ot_n = thr.chunks_with_hop(other[c])
        mixed = np.array([rv + ot if f else yy for yy, rv, ot, f in zip(y_n, rv_n, ot_n, flags)])
        comb = thr.concat_with_hop(mixed)
        if len(comb) < T:  # Rest hinter dem letzten Fenster: 20 ms Crossfade
            L0, fl = len(comb), int(0.02 * sr)
            src = (r_voice[c] + other[c]) if tail_speech else y[c]
            fi, fo = np.linspace(0, 1, fl), np.linspace(1, 0, fl)
            comb = np.concatenate((comb[:-fl], comb[-fl:] * fo + src[L0 - fl: L0] * fi, src[L0:]))
        out[c] = comb[:T]

    info = {"plan": plan, "speech_windows": flags, "voice": voice, "other": other}
    return out, info