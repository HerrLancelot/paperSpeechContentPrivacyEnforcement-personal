#!/usr/bin/env python
"""
Nutzwert-Auswertung (bleiben die Umgebungsgeraeusche erhalten?) fuer die Anonymisierungspipeline.

1) FAD wie im Paper: PANNs wavegram-logmel (Modellklasse aus embs/model_loader.py), Frame-Embeddings
   aller Dateien einer Variante gepoolt, Frechet-Distanz zwischen den Verteilungen.
   Referenzen: oracle (Originalmischung, wie im Paper) UND background (reines Background-Target).
   utils/fad.py wird NICHT importiert (braucht hypy_utils und `from scipy import sqrt`, das es in
   aktuellem SciPy nicht mehr gibt); die Frechet-Distanz ist hier mathematisch dieselbe, aber stabiler berechnet.

2) (nur mit --with-beats) BEATs, label-frei: Das Paper misst die Klassifikationsgenauigkeit gegen AudioSet-Labels der SONYC-Clips.
   Dieses Dataset hat keine AudioSet-Labels pro Hintergrundquelle, daher werden die BEATs-Ausgaben auf dem
   reinen Background-Target als Pseudo-Referenz benutzt: Wie gut stimmen die Nicht-Sprach-Klassen von
   Output (und Originalmischung, und Rauschen) mit denen des Background-Targets ueberein?

Aufruf (aus dem Repo-Root, cuDNN-Zeile vorher setzen):
  python eval_utility.py \
    --meta /pfad/test_metadata.jsonl \
    --orig-dir datasets/dataset_full/test \
    --sys nbss_ep36=/project/brunke/nbss_runs/test_out \
    --out-dir /project/brunke/nbss_runs/utility
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16000
T = 10.0


# ----------------------------------------------------------------------------- reine Rechenfunktionen
def frechet(mu1, c1, mu2, c2):
    """Frechet-Distanz zweier Gauss-Verteilungen. Spur der Matrixwurzel ueber symmetrische Eigenwerte
    (sqrt(C1) C2 sqrt(C1) ist symmetrisch PSD), dadurch reell und stabil auch bei Rangdefizit."""
    diff = mu1 - mu2
    w, v = np.linalg.eigh(c1)
    s = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = s @ c2 @ s
    ev = np.linalg.eigvalsh((m + m.T) / 2)
    tr_sqrt = np.sqrt(np.clip(ev, 0, None)).sum()
    return float(diff @ diff + np.trace(c1) + np.trace(c2) - 2 * tr_sqrt)


def stats(frames):
    frames = np.asarray(frames, dtype=np.float64)
    return frames.mean(0), np.cov(frames, rowvar=False)


def speech_indices(label_dict, ontology):
    """Indizes aller BEATs-Klassen im AudioSet-Teilbaum 'Speech' (Speech, Conversation, Narration, ...)."""
    by_id = {e["id"]: e for e in ontology}
    root = next(e["id"] for e in ontology if e["name"] == "Speech")
    seen, stack = {root}, [root]
    while stack:
        for c in by_id[stack.pop()].get("child_ids", []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return sorted(int(k) for k, v in label_dict.items() if v in seen)


def beats_metrics(p, p_ref, speech_idx):
    """p, p_ref: (N, 527) Wahrscheinlichkeiten. Rueckgabe pro Sample: cos, top1, recall5, speech_max."""
    keep = np.setdiff1d(np.arange(p.shape[1]), speech_idx)
    a, b = p[:, keep], p_ref[:, keep]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    top1 = (a.argmax(1) == b.argmax(1)).astype(float)
    ta = np.argsort(-a, axis=1)[:, :5]
    tb = np.argsort(-b, axis=1)[:, :5]
    rec5 = np.array([len(set(x) & set(y)) / 5 for x, y in zip(ta, tb)])
    return {"cos": cos, "top1": top1, "recall5": rec5, "speech_max": p[:, speech_idx].max(1)}


# ----------------------------------------------------------------------------- Audio / Modelle
def load_ch0(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: {sr} Hz"
    return x[:, 0].copy()


def white_noise(idx):
    n = np.random.default_rng(idx).normal(0, 1, int(SR * T))
    return np.clip(n, -1, 1).astype(np.float32)  # wie sf.write(PCM16) im Paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--sys", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--skip-fad", action="store_true")
    ap.add_argument("--with-beats", action="store_true",
                    help="zusaetzlich die label-freie BEATs-Auswertung (Standard: nur FAD; die BEATs-Genauigkeit "
                         "des Papers kommt aus eval_beats_accuracy.py)")
    ap.add_argument("--skip-beats", action="store_true", help="veraltet, ohne Wirkung (BEATs ist jetzt aus)")
    ap.add_argument("--beats-ckpt", default="./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    ap.add_argument("--ontology", default="./utils/audioset_ontology.json")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.25)
    a = ap.parse_args()

    import librosa
    import torch
    from scipy.stats import binomtest, ttest_rel

    sys.path.insert(0, os.getcwd())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and a.gpu_mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = [json.loads(l) for l in open(a.meta)]
    ids = [m["sample_id"] for m in meta]
    snr = {m["sample_id"]: m["achieved_snr_db"] for m in meta}

    orig = Path(a.orig_dir)
    variants = {  # Name -> sample_id, idx -> Audio (Kanal 0, 16 kHz)
        "oracle": lambda i, s: load_ch0(orig / f"{s}_mixture.wav"),
        "background": lambda i, s: load_ch0(orig / f"{s}_background.wav"),
    }
    if not a.no_noise:
        variants["noise"] = lambda i, s: white_noise(i)
    sysnames = []
    for item in a.sys:
        name, d = item.split("=", 1)
        variants[name] = (lambda d: lambda i, s: load_ch0(Path(d) / f"{s}_mixture.wav"))(d)
        sysnames.append(name)
        ids = [s for s in ids if (Path(d) / f"{s}_mixture.wav").exists()]  # nur Samples, die ueberall vorliegen
    print(f"{len(ids)} Samples, Varianten: {list(variants)}")
    idx_of = {m["sample_id"]: i for i, m in enumerate(meta)}

    # ------------------------------------------------------------------------- FAD
    if not a.skip_fad:
        from embs.model_loader import PANNsModel
        panns = PANNsModel(variant="wavegram-logmel", device=device)
        panns.load_model()
        frames = {}
        for v, fn in variants.items():
            d = out / "emb" / v
            d.mkdir(parents=True, exist_ok=True)
            embs = []
            for s in ids:
                f = d / f"{s}.npy"
                if not f.exists():
                    y = fn(idx_of[s], s)
                    y = librosa.resample(y, orig_sr=SR, target_sr=panns.sr)  # PANNs-Variante laeuft mit 32 kHz
                    y = y[None, :] / max(float(np.abs(y).max()), 1e-9)  # Normalisierung wie utils/fad.py
                    np.save(f, panns.get_embedding(y))
                embs.append(np.load(f))
            frames[v] = np.concatenate(embs).astype(np.float64)
            print(f"[FAD] {v}: {frames[v].shape[0]} Frames", flush=True)
        st = {v: stats(x) for v, x in frames.items()}
        fad = {}
        for ref in ("oracle", "background"):
            for v in variants:
                if v != ref:
                    fad[f"{v} vs {ref}"] = frechet(*st[ref], *st[v])
        json.dump(fad, open(out / "fad.json", "w"), indent=1)
        print("\n=== FAD (PANNs wavegram-logmel, kleiner = naeher an der Referenz) ===")
        for ref in ("oracle", "background"):
            print(f"  Referenz {ref}:")
            for v in variants:
                if v != ref:
                    print(f"    {v:14s} {fad[f'{v} vs {ref}']:9.2f}")
        del panns, frames
        if device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------------- BEATs
    if a.with_beats:
        sys.path.insert(0, os.path.join(os.getcwd(), "beats"))
        from BEATs import BEATs, BEATsConfig
        ck = torch.load(os.path.abspath(a.beats_ckpt), map_location=device)
        model = BEATs(BEATsConfig(ck["cfg"]))
        model.load_state_dict(ck["model"])
        model.eval().to(device)
        sp_idx = speech_indices(ck["label_dict"], json.load(open(a.ontology)))
        print(f"\n[BEATs] {len(sp_idx)} Sprach-Klassen ausgeschlossen")

        probs = {}
        for v, fn in variants.items():
            f = out / f"beats_probs_{v}.npy"
            if f.exists() and np.load(f).shape[0] == len(ids):
                probs[v] = np.load(f)
                continue
            rows = []
            for s in ids:
                x = torch.from_numpy(fn(idx_of[s], s)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pm = torch.zeros(x.shape, dtype=torch.bool, device=device)
                    rows.append(model.extract_features(x, padding_mask=pm)[0][0].cpu().numpy())
            probs[v] = np.stack(rows)
            np.save(f, probs[v])
            print(f"[BEATs] {v} fertig", flush=True)

        ref = probs["background"]
        rec, per = [], {}
        for v in variants:
            if v == "background":
                continue
            m = beats_metrics(probs[v], ref, sp_idx)
            per[v] = m
            for k, s in enumerate(ids):
                rec.append({"sample_id": s, "snr": snr[s], "variant": v, **{kk: float(vv[k]) for kk, vv in m.items()}})
        df = pd.DataFrame(rec)
        df.to_csv(out / "beats_per_sample.csv", index=False)

        print("\n=== BEATs-Szenenerhalt gegen das Background-Target (ohne Sprach-Klassen) ===")
        print("  cos = Kosinus der Nicht-Sprach-Wahrscheinlichkeiten, top1 = gleiche wahrscheinlichste Klasse,")
        print("  recall5 = Anteil der Top-5-Klassen des Targets im Top-5 der Variante, speech = max. Sprach-Klasse\n")
        for v, m in per.items():
            print(f"  {v:14s} cos {m['cos'].mean():.3f} | top1 {100 * m['top1'].mean():5.1f} % | "
                  f"recall5 {100 * m['recall5'].mean():5.1f} % | speech {m['speech_max'].mean():.3f}")
        bins = sorted({int(np.floor(x / 5) * 5) for x in snr.values()})
        print("\n  nach SNR (cos / top1 %):")
        for b in bins:
            sel = np.array([b <= snr[s] < b + 5 for s in ids])
            cells = "  ".join(f"{v}={per[v]['cos'][sel].mean():.3f}/{100 * per[v]['top1'][sel].mean():4.0f}" for v in per)
            print(f"    [{b:3d},{b + 5:3d}) n={sel.sum():3d}: {cells}")

        for name in sysnames:
            if "oracle" not in per:
                continue
            _, p = ttest_rel(per[name]["cos"], per["oracle"]["cos"])
            b_ = int(((per[name]["top1"] == 1) & (per["oracle"]["top1"] == 0)).sum())
            c_ = int(((per[name]["top1"] == 0) & (per["oracle"]["top1"] == 1)).sum())
            pm = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ > 0 else 1.0
            print(f"\n  {name} vs oracle: cos paired-t p={p:.4f} | top1 McNemar p={pm:.4f} (nur {name} richtig: {b_}, nur oracle richtig: {c_})")


if __name__ == "__main__":
    main()