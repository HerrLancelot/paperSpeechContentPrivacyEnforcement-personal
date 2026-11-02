#!/usr/bin/env python
"""
eval_beats_accuracy.py -- BEATs-Genauigkeitsverlust nach Tailleur et al. auf dem SCAD-Eval-Set

Paper-Metrik (compute_beats.py + metric_accu_drop.py), hier 1:1 uebernommen:
  - BEATs-Wahrscheinlichkeiten nur fuer 8 Klassen: Speech + die 7 SONYC-Zielklassen
  - Mischung (Sprache + Hintergrund): Top-2 der 8 Klassen, jede Vorhersage ist richtig, wenn sie 'Speech' ODER dem
    Hintergrundlabel entspricht
  - Kontrolle (nur Hintergrund, "Gruppe 0"): Top-1 der 8 Klassen, richtig nur beim Hintergrundlabel
  - Genauigkeit gepoolt ueber alle Vorhersagen, Clopper-Pearson-Intervall (alpha = 0.01, Halbbreite),
    Verlust = Genauigkeit(oracle) - Genauigkeit(System), McNemar (exakt) auf den gepaarten Vorhersagen
Zusaetzlich (nicht aus dem Paper): "Hintergrund-Treffer@1" -- ist die wahrscheinlichste der 7 Nicht-Sprach-Klassen
das Hintergrundlabel? Das misst direkt, ob die Umgebungsgeraeusche erhalten bleiben, unabhaengig davon, dass bei
einer Mischung ein Top-2-Platz ohne Sprache nie "richtig" sein kann.

Interpretation: Waveform-Reversal macht die Woerter unverstaendlich, BEATs erkennt aber weiter "Sprache" als
Klasse. Eine hohe Paper-Genauigkeit der Mischungen bedeutet daher: Szene inklusive Sprachpraesenz bleibt erhalten
(das ist beim Reversal gewollt, die Verstaendlichkeit misst die WER). Nur wenn Sprache komplett aus dem BEATs-Ranking
verschwindet (z. B. Unterdrueckung statt Reversal), faellt eine Mischung auf hoechstens 50 %, weil der Speech-Platz
dann nie richtig sein kann. Die Zusatzmetriken (Zerlegung, Hintergrund-Treffer@1, strict) sind NICHT aus dem Paper;
fuer die Vergleichbarkeit zaehlt allein der Block "Paper-Metrik (SCAD)".

Aufruf (aus dem Repo-Root, cuDNN-Zeile vorher setzen):
  python eval_beats_accuracy.py \
    --labels-csv dataset_out/scad_eval_labels.csv \
    --orig-dir dataset_out/scad_eval \
    --sys nbss_ep36=/project/brunke/nbss_runs/scad_mix_out,/project/brunke/nbss_runs/scad_ctrl_out \
    --out-dir /project/brunke/nbss_runs/scad_beats
  (--sys NAME=MIXDIR,CTRLDIR; ohne CTRLDIR nur Mischungen. Ausgaben heissen wie die Eingaben:
   <id>_mixture.wav bzw. <id>_control.wav)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16000
CLASSES = ["Speech", "Engine", "Jackhammer", "Chainsaw", "Vehicle horn, car horn, honking", "Siren", "Music", "Dog"]


# ----------------------------------------------------------------------------- reine Rechenfunktionen
def paper_predictions(p8, label, kind):
    """Paper-Regel: mix -> Top-2 (Speech oder Label richtig), ctrl -> Top-1 (nur Label richtig)."""
    top = np.argsort(-np.asarray(p8))[: 2 if kind == "mix" else 1]
    return [int(CLASSES[j] == label or (kind == "mix" and CLASSES[j] == "Speech")) for j in top]


def bg_hit(p8, label):
    """Hintergrund-Treffer@1 ueber die 7 Nicht-Sprach-Klassen."""
    return int(CLASSES[1:][int(np.argmax(np.asarray(p8)[1:]))] == label)


def top2_flags(p8, label):
    """(Hintergrundlabel in Top-2, Speech in Top-2). Fuer Mischungen gilt: Paper-Genauigkeit = Mittel beider Werte."""
    top = [CLASSES[j] for j in np.argsort(-np.asarray(p8))[:2]]
    return int(label in top), int("Speech" in top)


def clopper_pearson(k, n, alpha=0.01):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def mcnemar_exact(a, b):
    """a, b: gepaarte 0/1-Listen. Rueckgabe (p, nur a richtig, nur b richtig)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a), np.asarray(b)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0), n10, n01


def acc_str(x):
    x = np.asarray(x)
    k, n = int(x.sum()), len(x)
    lo, hi = clopper_pearson(k, n)
    return f"{100 * k / n:5.1f} +-{50 * (hi - lo):4.1f} (n={n})"


def evaluate(df, probs, variants, ref="oracle"):
    """df: Label-CSV; probs[variant] = {"mix": (N,8), "ctrl": (N,8) oder None}. Gibt nichts zurueck, druckt Tabellen."""
    labels = df["label"].tolist()
    has_ctrl = all(probs[v].get("ctrl") is not None for v in variants)
    kinds = ["mix"] + (["ctrl"] if has_ctrl else [])

    def preds(v, mask):
        out = []
        for kind in kinds:
            for i in np.where(mask)[0]:
                out += paper_predictions(probs[v][kind][i], labels[i], kind)
        return np.array(out)

    def bg(v, kind, mask):
        return np.array([bg_hit(probs[v][kind][i], labels[i]) for i in np.where(mask)[0]])

    all_mask = np.ones(len(df), bool)
    strict_mask = df["strict"].astype(str).str.lower().eq("true").to_numpy()

    for title, mask in (("alle Szenen", all_mask), ("nur strict-Clips", strict_mask)):
        if mask.sum() == 0:
            continue
        print(f"\n=== Paper-Metrik (SCAD): Top-2/Top-1-Genauigkeit ({title}, {mask.sum()} Szenen"
              f"{', mit Kontrollen' if has_ctrl else ', ohne Kontrollen'}) ===")
        acc_ref = preds(ref, mask).mean()
        for v in variants:
            p = preds(v, mask)
            lo, hi = clopper_pearson(int(p.sum()), len(p))
            print(f"  {v:14s} {acc_str(p)}   Accuracy-Drop gegen {ref}: {100 * (acc_ref - p.mean()):5.1f} +-{50 * (hi - lo):4.1f}")
        for v in variants:
            if v != ref:
                pv, b, c = mcnemar_exact(preds(v, mask), preds(ref, mask))
                print(f"  McNemar {v} vs {ref}: p = {pv:.4f} (nur {v} richtig: {b}, nur {ref} richtig: {c})")

        systems = [v for v in variants if v not in (ref, "noise")]
        if len(systems) >= 2:
            best = max(systems, key=lambda v: preds(v, mask).mean())
            print(f"  Bestes System (hoechste Genauigkeit): {best}")
            for v in systems:
                if v != best:
                    pv, b, c = mcnemar_exact(preds(v, mask), preds(best, mask))
                    sig = "NICHT signifikant" if pv > 0.01 else "signifikant"
                    print(f"  McNemar {best} vs {v}: p = {pv:.4f} (b={b}, c={c}) -> {sig}")

        print(f"\n  Zusatz (nicht im Paper) -- Zerlegung der Mischungen (Paper-Genauigkeit der Mischungen = Mittel aus beiden Zeilen):")
        print(f"  {'':14s}{'Hintergrund in Top-2':>26s}{'Sprache in Top-2':>22s}")
        for v in variants:
            fl = np.array([top2_flags(probs[v]["mix"][i], labels[i]) for i in np.where(mask)[0]])
            print(f"  {v:14s}{acc_str(fl[:, 0]):>26s}{acc_str(fl[:, 1]):>22s}")
        if has_ctrl:
            print("  Kontrollen (Top-1 = Hintergrundlabel):")
            for v in variants:
                ck_ = np.array([paper_predictions(probs[v]["ctrl"][i], labels[i], "ctrl")[0] for i in np.where(mask)[0]])
                print(f"    {v:14s} {acc_str(ck_)}")

        print(f"\n=== Zusatz (nicht im Paper): Hintergrund-Treffer@1 (7 Nicht-Sprach-Klassen, {title}) ===")
        for kind in kinds:
            print(f"  {'Mischungen' if kind == 'mix' else 'Kontrollen (ohne Sprache)'}:")
            for v in variants:
                print(f"    {v:14s} {acc_str(bg(v, kind, mask))}")
            for v in variants:
                if v != ref:
                    pv, b, c = mcnemar_exact(bg(v, kind, mask), bg(ref, kind, mask))
                    print(f"    McNemar {v} vs {ref}: p = {pv:.4f} (nur {v}: {b}, nur {ref}: {c})")

    print("\n=== Hintergrund-Treffer@1 je Label (Mischungen, alle Szenen) ===")
    labs = sorted(set(labels))
    print("  " + f"{'':32s}" + "".join(f"{v[:12]:>14s}" for v in variants))
    for lab in labs:
        m = np.array([l == lab for l in labels])
        print("  " + f"{lab[:30]:32s}" + "".join(f"{100 * bg(v, 'mix', m).mean():10.1f} %  " for v in variants)
              + f"  (n={m.sum()})")


# ----------------------------------------------------------------------------- Audio / BEATs
def load_ch0(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: {sr} Hz"
    return x[:, 0].copy()


def white_noise(seed):
    n = np.random.default_rng(seed).normal(0, 1, SR * 10)
    return np.clip(n, -1, 1).astype(np.float32)  # wie sf.write(PCM16) im Paper


def beats_probs8(model, device, wavs, idx):
    import torch
    rows = []
    for w in wavs:
        x = torch.from_numpy(w).unsqueeze(0).to(device)
        with torch.no_grad():
            pm = torch.zeros(x.shape, dtype=torch.bool, device=device)
            rows.append(model.extract_features(x, padding_mask=pm)[0][0].cpu().numpy()[idx])
    return np.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--sys", action="append", default=[], metavar="NAME=MIXDIR[,CTRLDIR]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--beats-ckpt", default="./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    ap.add_argument("--ontology", default="./utils/audioset_ontology.json")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.25)
    a = ap.parse_args()

    import torch
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "beats"))
    from BEATs import BEATs, BEATsConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and a.gpu_mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.labels_csv)
    unknown = set(df["label"]) - set(CLASSES)
    assert not unknown, f"Labels ausserhalb der 8 Klassen: {unknown}"
    ids = df["sample_id"].tolist()
    orig = Path(a.orig_dir)

    # Quellen je Variante: kind -> Funktion(i, id) -> Audio
    src = {"oracle": {"mix": lambda i, s: load_ch0(orig / f"{s}_mixture.wav"),
                      "ctrl": lambda i, s: load_ch0(orig / f"{s}_control.wav")}}
    if not a.no_noise:
        src["noise"] = {"mix": lambda i, s: white_noise(2 * i), "ctrl": lambda i, s: white_noise(2 * i + 1)}
    for item in a.sys:
        name, dirs = item.split("=", 1)
        mixdir, *rest = dirs.split(",")
        src[name] = {"mix": (lambda d: lambda i, s: load_ch0(Path(d) / f"{s}_mixture.wav"))(mixdir),
                     "ctrl": (lambda d: lambda i, s: load_ch0(Path(d) / f"{s}_control.wav"))(rest[0]) if rest else None}
    variants = list(src)
    if any(src[v]["ctrl"] is None for v in variants):
        print("Mindestens ein System ohne Kontroll-Ordner -> Kontrollen werden fuer ALLE Varianten ausgelassen.")
        for v in variants:
            src[v]["ctrl"] = None

    ck = torch.load(os.path.abspath(a.beats_ckpt), map_location=device)
    model = BEATs(BEATsConfig(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    name_to_id = {e["name"]: e["id"] for e in json.load(open(a.ontology))}
    inv = {v: int(k) for k, v in ck["label_dict"].items()}
    idx = [inv[name_to_id[c]] for c in CLASSES]
    print("BEATs-Klassenindizes:", dict(zip(CLASSES, idx)))

    probs = {}
    for v in variants:
        probs[v] = {}
        for kind in ("mix", "ctrl"):
            if src[v][kind] is None:
                probs[v][kind] = None
                continue
            f = out / f"probs8_{v}_{kind}.npy"
            if f.exists() and np.load(f).shape[0] == len(ids):
                probs[v][kind] = np.load(f)
                continue
            probs[v][kind] = beats_probs8(model, device, (src[v][kind](i, s) for i, s in enumerate(ids)), idx)
            np.save(f, probs[v][kind])
            print(f"[BEATs] {v}/{kind} fertig", flush=True)

    evaluate(df, probs, variants)


if __name__ == "__main__":
    main()