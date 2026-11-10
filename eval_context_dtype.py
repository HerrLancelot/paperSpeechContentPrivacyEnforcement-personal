"""Offline-Vortest fuer die Fallback-Stufen aus Abschnitt 6: kuerzerer Kontext und fp16 (ohne VAD/Splicing).

Faehrt NUR den Separator blockweise wie im Live-Betrieb (Block 2,26 s + Links-Kontext, Block 0 ohne Kontext) und
vergleicht die Sprach-Schaetzung
  (a) gegen die Ganzdatei-Schaetzung in fp32 (= unendlicher Kontext, Referenz des Gleichheitstests),
  (b) bei fp16/bf16 gegen dieselbe Kontextlaenge in fp32 (reiner Rundungsanteil),
  (c) optional per SI-SDR gegen ein sauberes Sprach-Target (--target, Kanal 0).
Dazu: Abweichung ueber die Position im Block (5 Fuenftel; zeigt, ob der Blockanfang unter kurzem Kontext leidet),
Zeit je Block und fp16-Ueberlaeufe. Block 0 faellt aus der Statistik (hat in jeder Variante keinen Kontext).
Die Qualitaetszahlen sind hardwareunabhaengig (Cluster/Laptop genuegt); die Zeiten zaehlen nur auf der Ziel-GPU.

  python eval_context_dtype.py --ckpt <ckpt> --wav <4ch_16k.wav> --device cuda --seconds 60 \
         --ctx 2,1,0.5 --dtypes fp32,fp16,bf16 [--target clean_speech.wav] [--arch retention|mamba] --json eval.json

Richtwerte (Gleichheitstest): >60 dB praktisch identisch; bf16 lag bei 44 bis 49 dB. fp16 sollte darueber liegen,
solange es keine Ueberlaeufe gibt (Spalte 'fb' = Fallbacks nach fp32 muss 0 sein).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nbss_separator import NBSSSeparator   # noqa: E402
from stream_anonymizer import snap_block   # noqa: E402


def rel_db(ref, est):
    ref, est = np.asarray(ref, np.float64), np.asarray(est, np.float64)
    return 10 * np.log10((ref ** 2).sum() / max(((ref - est) ** 2).sum(), 1e-30))


def si_sdr(target, est):
    t, e = np.asarray(target, np.float64), np.asarray(est, np.float64)
    t, e = t - t.mean(), e - e.mean()
    proj = (e @ t) / max(t @ t, 1e-30) * t
    return 10 * np.log10(max((proj ** 2).sum(), 1e-30) / max(((e - proj) ** 2).sum(), 1e-30))


def run_blockwise(sep, x, block, ctx_s, sr, warm=2):
    """Wie StreamAnonymizer/ContextSeparator: je Block [Links-Kontext + Block] -> nur der Blockanteil zaehlt."""
    nb, c = x.shape[1] // block, int(round(ctx_s * sr))
    sp, bg, times = np.zeros(nb * block, np.float32), np.zeros(nb * block, np.float32), []
    for b in range(-warm, nb):               # negative Indizes: Warm-up (CUDA-Init), nicht gezaehlt
        s = max(b, 0) * block
        seg = np.ascontiguousarray(x[:, max(0, s - c):s + block])
        t0 = time.perf_counter()
        v, o = sep.separate(seg)             # .cpu().numpy() synchronisiert die GPU
        dt = time.perf_counter() - t0
        if b >= 0:
            sp[s:s + block], bg[s:s + block] = v[-block:], o[-block:]
            times.append(dt)
    return sp, bg, np.array(times)


def by_position(ref, est, nb, block, k=5):
    r, e = ref[:nb * block].reshape(nb, block)[1:], est[:nb * block].reshape(nb, block)[1:]
    w = block // k
    return [rel_db(r[:, j * w:(j + 1) * w], e[:, j * w:(j + 1) * w]) for j in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", required=True, help="4 Kanaele, 16 kHz (z. B. Raumaufnahme oder synthetischer Clip mit Sprache)")
    ap.add_argument("--target", help="optional: sauberes Sprach-Target (wav, Kanal 0) fuer SI-SDR; am besten dasselbe Target wie im Training")
    ap.add_argument("--arch", choices=["retention", "mamba"], default="retention")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ctx", default="2,1,0.5", help="Kontextlaengen in s, kommagetrennt")
    ap.add_argument("--dtypes", default="fp32,fp16,bf16")
    ap.add_argument("--block", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--json")
    args = ap.parse_args()

    sep = NBSSSeparator(args.ckpt, device=args.device, precision="fp32", seg=None, arch=args.arch)
    sr, block = sep.sr, snap_block(args.block, sep.sr)
    x, fs = sf.read(args.wav, dtype="float32", always_2d=True)
    assert fs == sr and x.shape[1] >= sep.N_CHN, (fs, x.shape)
    x = x.T[:sep.N_CHN, :int(args.seconds * sr)]
    nb = x.shape[1] // block
    assert nb >= 3, "zu kurz: --seconds erhoehen oder laengere Datei"
    x = x[:, :nb * block]
    tgt = None
    if args.target:
        tgt, ft = sf.read(args.target, dtype="float32", always_2d=True)
        assert ft == sr and tgt.shape[0] >= nb * block, (ft, tgt.shape)
        tgt = tgt[:nb * block, 0]

    ctxs = [float(c) for c in args.ctx.split(",")]
    dtypes = [d for d in args.dtypes.split(",")]
    if sep.device.type != "cuda" and dtypes != ["fp32"]:
        print("[i] CPU: nur fp32 moeglich")
        dtypes = ["fp32"]
    if "fp32" not in dtypes:
        dtypes.insert(0, "fp32")             # fp32-Lauf je Kontext ist die Basis fuer (b)
    dtypes.sort(key=lambda d: d != "fp32")

    print(f"Geraet={sep.device}, {nb} Bloecke a {block} Samples = {block / sr:.3f} s ({nb * block / sr:.1f} s), arch={args.arch}")
    sep.precision = "fp32"
    ref_sp, ref_bg = sep.separate(x)         # Ganzdatei, fp32
    ref_sp, ref_bg = ref_sp[:nb * block], ref_bg[:nb * block]
    if tgt is not None:
        print(f"SI-SDR Ganzdatei fp32 (Basis): {si_sdr(tgt, ref_sp):.2f} dB")
    print(f"\n{'ctx s':>5} {'dtype':>5} | {'vs Ganzdatei':>12} {'Hintergr.':>9} | {'vs fp32 (gl. ctx)':>17} | "
          f"{'dB je Blockfuenftel (Anfang -> Ende)':<40} | {'SI-SDR':>7} | {'Median':>6} {'p95':>6} {'max':>6} ms {'RTFmax':>6} | fb")
    results, base = [], {}
    for ctx in ctxs:
        for dt in dtypes:
            sep.precision, fb0 = dt, sep.n_fp16_fallback
            sp, bg, times = run_blockwise(sep, x, block, ctx, sr)
            fb = sep.n_fp16_fallback - fb0
            if dt == "fp32":
                base[ctx] = sp
            ok = bool(np.isfinite(sp).all())
            d_ref = rel_db(ref_sp[block:], sp[block:])
            d_bg = rel_db(ref_bg[block:], bg[block:])
            d_fp32 = rel_db(base[ctx][block:], sp[block:]) if dt != "fp32" else float("nan")
            pos = by_position(ref_sp, sp, nb, block)
            sisdr = si_sdr(tgt, sp) if tgt is not None else float("nan")
            ms = times * 1000
            r = dict(ctx_s=ctx, dtype=dt, db_vs_whole=d_ref, db_bg_vs_whole=d_bg, db_vs_fp32_same_ctx=d_fp32,
                     db_by_position=pos, si_sdr=sisdr, ms_median=float(np.median(ms)), ms_p95=float(np.percentile(ms, 95)),
                     ms_max=float(ms.max()), rtf_max=float(times.max() / (block / sr)), fp16_fallbacks=fb, finite=ok)
            results.append(r)
            print(f"{ctx:5.2f} {dt:>5} | {d_ref:10.1f}dB {d_bg:7.1f}dB | {d_fp32:15.1f}dB | "
                  f"{'  '.join(f'{v:5.1f}' for v in pos):<40} | {sisdr:7.2f} | {r['ms_median']:6.0f} {r['ms_p95']:6.0f} "
                  f"{r['ms_max']:6.0f}    {r['rtf_max']:6.2f} | {fb}{'' if ok else '  NICHT ENDLICH!'}")
    print("\nLesehilfe: 'vs Ganzdatei' = Verlust durch begrenzten Kontext (+ Rundung); 'vs fp32' = nur Rundung; "
          "ein Abfall im ersten Fuenftel zeigt, dass der Blockanfang unter kurzem Kontext leidet.\n"
          "Zeiten gelten nur auf der Ziel-GPU (hier gemessen auf: %s)." % (torch.cuda.get_device_name(0) if sep.device.type == "cuda" else "CPU"))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(block_s=block / sr, arch=args.arch, wav=args.wav, results=results), f, indent=1)


if __name__ == "__main__":
    main()