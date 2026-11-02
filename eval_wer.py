#!/usr/bin/env python
# eval_wer.py (v2)
import argparse, csv, json
import numpy as np, soundfile as sf, whisper, jiwer
from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--orig-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--hyps", required=True)       # jsonl mit allen Hypothesen
ap.add_argument("--model", default="medium.en")
a = ap.parse_args()

norm = EnglishTextNormalizer()
model = whisper.load_model(a.model)
T = 10.0

def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    assert sr == 16000, sr
    return x[:, 0]

def asr(p):  # Default-Temperaturfallback bleibt aktiv
    r = model.transcribe(load(p), language="en", fp16=True,
                         condition_on_previous_text=False)
    return norm(r["text"])

def score(ref, hyp):
    n = len(ref.split())
    if n == 0:
        return dict(n=0, err=0, hit=0)
    if not hyp.strip():
        return dict(n=n, err=n, hit=0)
    o = jiwer.process_words(ref, hyp)
    return dict(n=n, err=o.substitutions + o.deletions + o.insertions, hit=o.hits)

def complete(m):  # passt jede Sprachquelle komplett in die 10 s?
    try:
        return all(s["delay"] + sf.info(s["file"]).duration <= T
                   for s in m["sources"] if s["role"] == "speech")
    except Exception:
        return None  # FLAC nicht erreichbar

rows = []
with open(a.hyps, "w") as fh:
    for line in open(a.meta):
        m = json.loads(line)
        sid = m["sample_id"]
        out_p = Path(a.out_dir) / f"{sid}_mixture.wav"
        if not out_p.exists():
            continue
        sp = sorted((s for s in m["sources"] if s["role"] == "speech"),
                    key=lambda s: s["turn_idx"] or 0)
        ref = norm(" ".join(s["text"] for s in sp))
        paths = {"mix": Path(a.orig_dir) / f"{sid}_mixture.wav", "out": out_p,
                 "tgt": Path(a.orig_dir) / f"{sid}_speech.wav"}
        hyp = {k: asr(p) for k, p in paths.items()}
        fh.write(json.dumps({"sample_id": sid, "ref": ref,
                             **{f"hyp_{k}": v for k, v in hyp.items()}}) + "\n")
        fh.flush()

        row = {"sample_id": sid, "snr": m["achieved_snr_db"], "complete": complete(m)}
        for k in ("mix", "out", "tgt"):                       # Referenz: LibriSpeech-Text
            s = score(ref, hyp[k])
            row["n_lib"] = s["n"]; row[f"err_lib_{k}"] = s["err"]; row[f"hit_lib_{k}"] = s["hit"]
        row["n_tg"] = 0                                       # Referenz: Whisper auf Target
        for k in ("mix", "out"):
            s = score(hyp["tgt"], hyp[k])
            row["n_tg"] = s["n"]; row[f"err_tg_{k}"] = s["err"]; row[f"hit_tg_{k}"] = s["hit"]
        rows.append(row)
        print(sid, row["complete"], {k: v for k, v in row.items() if k.startswith("err")}, flush=True)

with open(a.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def summ(rs, mode, k):
    n = sum(r[f"n_{mode}"] for r in rs)
    if n == 0:
        return "-"
    err = sum(r[f"err_{mode}_{k}"] for r in rs)
    cap = sum(min(r[f"err_{mode}_{k}"], r[f"n_{mode}"]) for r in rs)
    hit = sum(r[f"hit_{mode}_{k}"] for r in rs)
    return f"WER {100*err/n:6.1f} | WER(cap) {100*cap/n:5.1f} | Treffer {100*hit/n:5.1f} %"

def report(title, rs, mode, keys):
    print(f"\n=== {title} (n={len(rs)}) ===")
    for k in keys:
        print(f"  gesamt {k:3s}: {summ(rs, mode, k)}")
    bins = sorted({int(np.floor(r['snr'] / 5) * 5) for r in rs})
    for b in bins:
        sub = [r for r in rs if int(np.floor(r['snr'] / 5) * 5) == b]
        print(f"  SNR [{b},{b+5}) n={len(sub)}")
        for k in keys:
            print(f"      {k:3s}: {summ(sub, mode, k)}")

full = [r for r in rows if r["complete"]]
report("LibriSpeech-Referenz, nur vollständige Samples", full, "lib", ["mix", "out", "tgt"])
report("Whisper-Target als Referenz, alle Samples", rows, "tg", ["mix", "out"])
print(f"\nvollständig: {len(full)}, unvollständig: {sum(r['complete'] is False for r in rows)}, "
      f"unbekannt: {sum(r['complete'] is None for r in rows)}")