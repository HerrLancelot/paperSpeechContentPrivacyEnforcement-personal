#!/usr/bin/env python
"""
WER-Auswertung nach Tailleur et al. (Segment-wise Waveform Reversal), angepasst auf das eigene Dataset.

Protokoll wie im Paper:
  - ASR-Systeme: Whisper large-v3, Wav2Vec2 (base-960h), Fairseq S2T, SpeechBrain CRDNN (asr-crdnn-rnnlm-librispeech)
  - Modell-Klassen und Normalisierung (jiwer-Transforms) kommen unveraendert aus voice_metrics.py
  - Referenz: LibriSpeech-Transkript, Datei-WER auf 1.0 gedeckelt
  - Mittelwert ueber alle Dateien UND alle ASR-Systeme, 95%-CI = 1.96 * std / sqrt(n)
  - Vergleichsbasis: oracle (Originalmischung) und noise (weisses Rauschen N(0,1), 10 s)

Abweichungen / Ergaenzungen:
  - Referenz bei Gespraechen: Texte aller Sprecher nach turn_idx verkettet
  - Zusatzauswertung nur ueber "vollstaendige" Samples (alle Aeusserungen passen in die 10 s)
  - Paired-t-Test auf dem Mittel ueber ASR-Systeme pro Sample
  - Hypothesen werden pro ASR gecacht (jsonl), ein Abbruch kostet nichts

Aufruf (aus dem Repo-Root, damit voice_metrics importierbar ist, cuDNN-Zeile vorher setzen):
  python eval_wer_paper.py \
    --meta /pfad/test_metadata.jsonl \
    --orig-dir datasets/dataset_full/test \
    --sys nbss_ep36=/project/brunke/nbss_runs/test_out \
    --out-dir /project/brunke/nbss_runs/wer_paper
"""
import argparse
import gc
import inspect
import json
import os
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import ttest_rel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, os.getcwd())
from voice_metrics import WerMetricSB, WerMetricSPT, WerMetricW2V2, WerMetricWhisper  # noqa: E402

SR = 16000
T_MAX = 10.0

# identisch zu den Transforms in voice_metrics.py
TRANS = jiwer.Compose([
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


class WhisperLarge(WerMetricWhisper):
    """voice_metrics.py hat whisper-tiny fest verdrahtet, das Paper nennt large-v3."""

    def __init__(self, device, name):
        super().__init__(device=device)  # laedt kurz whisper-tiny, wird sofort ersetzt
        del self.model
        gc.collect()
        self.processor = AutoProcessor.from_pretrained(name)
        # explizit fp32: neuere transformers-Versionen laden sonst den fp16-dtype des Checkpoints,
        # get_transcript() uebergibt aber fp32-Features
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()


def make_asr(name, device, whisper_name):
    if name == "whisper":
        return WhisperLarge(device, whisper_name)
    if name == "fairseq":
        return WerMetricSPT(device=device)
    if name == "w2v2":
        return WerMetricW2V2(device=device)
    if name == "crdnn":
        # neuere SpeechBrain-Versionen erwarten run_opts["device"] als String, nicht als torch.device
        return WerMetricSB(device=str(device), asr_short_name="cr")
    raise ValueError(name)


def load_wav(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: {sr} Hz"
    return torch.from_numpy(x[:, 0].copy()).unsqueeze(0)  # Kanal 0, Form (1, T)


def white_noise(idx):
    rng = np.random.default_rng(idx)
    n = rng.normal(0, 1, int(SR * T_MAX))
    return torch.from_numpy(np.clip(n, -1, 1).astype(np.float32)).unsqueeze(0)  # wie sf.write PCM16 im Paper


# jiwer 3.x/4.x: reference_transform, aeltere Versionen (Repo-Stand): truth_transform
_REF_KW = "reference_transform" if "reference_transform" in inspect.signature(jiwer.wer).parameters else "truth_transform"


def wer(ref, hyp):
    try:
        return jiwer.wer(ref, hyp, hypothesis_transform=TRANS, **{_REF_KW: TRANS})
    except ValueError:  # leere Hypothese, wie in voice_metrics.get_wer
        return 1.0


def read_meta(path):
    rows = []
    for line in open(path):
        m = json.loads(line)
        sp = sorted((s for s in m["sources"] if s["role"] == "speech"), key=lambda s: s["turn_idx"] or 0)
        try:
            complete = all(s["delay"] + sf.info(s["file"]).duration <= T_MAX for s in sp)
        except Exception:
            complete = False
        rows.append({"sample_id": m["sample_id"], "snr": m["achieved_snr_db"],
                     "ref": " ".join(s["text"] for s in sp), "complete": complete})
    return rows


def pooled(x):
    n = len(x)
    return 100 * x.mean(), 100 * 1.96 * x.std() / np.sqrt(n) if n > 1 else float("nan")


def report(df, title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")
    variants = list(dict.fromkeys(df["variant"]))
    n_s = df["sample_id"].nunique()
    print(f"{n_s} Samples, {df['asr'].nunique()} ASR-Systeme")
    means = {}
    for v in variants:
        sub = df[df["variant"] == v]
        m, ci = pooled(sub["wer"])
        means[v] = m
        per_asr = "  ".join(f"{a}={100 * g['wer'].mean():5.1f}" for a, g in sub.groupby("asr"))
        print(f"  {v:14s} WER {m:5.1f} +-{ci:4.1f}   | {per_asr}")

    bins = sorted({int(np.floor(s / 5) * 5) for s in df["snr"]})
    print("  nach SNR (gepoolt):")
    for b in bins:
        sel = df[(df["snr"] >= b) & (df["snr"] < b + 5)]
        cells = "  ".join(f"{v}={100 * sel[sel['variant'] == v]['wer'].mean():5.1f}" for v in variants)
        print(f"    [{b:3d},{b + 5:3d}) n={sel['sample_id'].nunique():3d}: {cells}")

    per_sample = df.groupby(["variant", "sample_id"])["wer"].mean().unstack(0)
    best = max(means, key=means.get)
    print(f"  Hoechste WER: {best}")
    for v in variants:
        if v == best:
            continue
        _, p = ttest_rel(per_sample[v], per_sample[best])
        sig = "NICHT signifikant" if p > 0.01 else "signifikant"
        print(f"    {best} vs {v}: p = {p:.4f} -> {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--orig-dir", required=True, help="Ordner mit {sample_id}_mixture.wav (Originalmischungen)")
    ap.add_argument("--sys", action="append", default=[], metavar="NAME=DIR",
                    help="anonymisierte Ausgabe, Dateien {sample_id}_mixture.wav; mehrfach angebbar")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--asr", nargs="+", default=["whisper", "fairseq", "w2v2", "crdnn"])
    ap.add_argument("--whisper-model", default="openai/whisper-large-v3")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--bg-baseline", action="store_true",
                    help="zusaetzliche Variante 'background': reines Background-Target {sample_id}_background.wav "
                         "aus --orig-dir (enthaelt garantiert keine Sprache, realistischere Untergrenze als weisses Rauschen)")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.25,
                    help="Deckel fuer GPU-Speicher dieses Prozesses (0 = aus), schuetzt das parallele Training")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and a.gpu_mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = read_meta(a.meta)
    print(f"{len(meta)} Samples, davon vollstaendig: {sum(m['complete'] for m in meta)}")

    # Variante -> Funktion(sample_idx, sample_id) -> Audio
    variants = {"oracle": lambda i, s: load_wav(Path(a.orig_dir) / f"{s}_mixture.wav")}
    if not a.no_noise:
        variants["noise"] = lambda i, s: white_noise(i)
    if a.bg_baseline:
        variants["background"] = lambda i, s: load_wav(Path(a.orig_dir) / f"{s}_background.wav")
    for item in a.sys:
        name, d = item.split("=", 1)
        variants[name] = (lambda d: lambda i, s: load_wav(Path(d) / f"{s}_mixture.wav"))(d)

    # fehlende Output-Dateien frueh melden
    for item in a.sys:
        name, d = item.split("=", 1)
        miss = [m["sample_id"] for m in meta if not (Path(d) / f"{m['sample_id']}_mixture.wav").exists()]
        if miss:
            print(f"WARNUNG {name}: {len(miss)} Dateien fehlen, werden uebersprungen (z. B. {miss[0]})")

    for asr in a.asr:
        cache_path = out / f"hyps_{asr}.jsonl"
        cache = {}
        if cache_path.exists():
            for line in open(cache_path):
                d = json.loads(line)
                cache[(d["variant"], d["sample_id"])] = d["hyp"]
        todo = [(v, i, m["sample_id"]) for v in variants for i, m in enumerate(meta)
                if (v, m["sample_id"]) not in cache]
        if not todo:
            continue
        print(f"\n[{asr}] lade Modell, {len(todo)} Transkriptionen offen", flush=True)
        model = make_asr(asr, device, a.whisper_model)
        fails = 0
        with open(cache_path, "a") as fh:
            for k, (v, i, sid) in enumerate(todo):
                try:
                    audio = variants[v](i, sid)
                except FileNotFoundError:
                    continue
                try:
                    hyp = model.get_transcript(audio, SR)
                    fails = 0
                except Exception as e:  # wie compute_wer.py: Fehler melden, nicht abbrechen
                    print(f"  Fehler {asr}/{v}/{sid}: {e}")
                    fails += 1
                    if fails >= 10:  # systematischer Fehler (z. B. dtype), nicht 960x durchlaufen
                        raise RuntimeError(f"{asr}: 10 Fehler in Folge, Abbruch") from e
                    continue
                cache[(v, sid)] = hyp
                fh.write(json.dumps({"variant": v, "sample_id": sid, "hyp": hyp}) + "\n")
                fh.flush()
                if k % 50 == 0:
                    print(f"  [{asr}] {k}/{len(todo)}", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Scoring aus dem Cache (getrennt von der Transkription, jederzeit wiederholbar)
    rows = []
    for asr in a.asr:
        cache_path = out / f"hyps_{asr}.jsonl"
        if not cache_path.exists():
            continue
        cache = {}
        for line in open(cache_path):
            d = json.loads(line)
            cache[(d["variant"], d["sample_id"])] = d["hyp"]
        for (v, sid), hyp in cache.items():
            m = next((x for x in meta if x["sample_id"] == sid), None)
            if m is None or v not in variants:
                continue
            raw = wer(m["ref"], hyp)
            rows.append({"asr": asr, "variant": v, "sample_id": sid, "snr": m["snr"],
                         "complete": m["complete"], "ref": m["ref"], "hyp": hyp,
                         "wer_raw": raw, "wer": min(raw, 1.0)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "all_results.csv", index=False)

    # nur Samples, fuer die jede Variante und jedes ASR-System vorliegt
    n_cells = df["variant"].nunique() * df["asr"].nunique()
    ok = df.groupby("sample_id").size()
    df = df[df["sample_id"].isin(ok[ok == n_cells].index)]

    report(df, "alle Samples")
    report(df[df["complete"]], "nur vollstaendige Samples (Aeusserungen passen in 10 s)")
    print(f"\nErgebnisse: {out / 'all_results.csv'}")


if __name__ == "__main__":
    main()