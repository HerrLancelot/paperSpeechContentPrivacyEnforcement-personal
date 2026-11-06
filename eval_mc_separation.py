"""
Punkt 2 (gerollte Referenz) und Hintergrund-TDOA-Teil von Punkt 4 in einem Durchlauf.

Fuer jede Szene im scad_eval-Set (mixture 4ch, control 4ch = gleiche Szene ohne Sprache):
  - Sprach-Referenz je Kanal:  speech_gt[c] = mixture[c] - control[c]   (setzt voraus, dass die
    Mischung exakt Hintergrund + Sprache ist; SI-SDR ist skalierungsinvariant, ein globaler Gain stoert nicht)
  - Fuer jede Rollung c=0..3: Separator mit np.roll(mix, -c) -> Sprach- und Background-Schaetzung am Kanal c
  - SI-SDR (Sprache vs speech_gt[c]), SI-SDRi gegenueber mixture[c], SI-SDR (Background vs control[c])
  - Hintergrund-Kohaerenz: GCC-PHAT-Lag zwischen Kanal 0 und Kanal c, in 1-s-Fenstern, fuer
        control (Wahrheit), Background-Schaetzung (aus 4 getrennten Laeufen), mixture (Baseline)
    Kennzahl: Anteil der Fenster mit |lag_est - lag_control| <= 1 Sample.

Aufruf (cuDNN-Zeile vorher setzen):
  python eval_mc_separation.py --scad-dir datasets/dataset_full/scad_eval --ckpt <ckpt> \
      --out /project/brunke/nbss_runs/mc_sep_eval --n 48 --seed 0
"""
import argparse
import csv
import glob
import os

import numpy as np
import soundfile as sf
import torch

from nbss_separator import NBSSSeparator

SR = 16000


def si_sdr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    t = (est @ ref) / (ref @ ref + eps) * ref
    n = est - t
    return 10 * np.log10((t @ t + eps) / (n @ n + eps))


def gcc_phat_lag(a, b, max_lag):
    n = len(a) + len(b)
    R = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    cc = np.fft.irfft(R / (np.abs(R) + 1e-12), n)
    cc = np.concatenate((cc[-max_lag:], cc[: max_lag + 1]))
    return int(np.argmax(cc) - max_lag)


def window_lags(x0, xc, win, max_lag, min_rms=1e-4):
    """Lag je Fenster, NaN wenn eines der Fenster (fast) still ist."""
    out = []
    for s in range(0, len(x0) - win + 1, win):
        a, b = x0[s:s + win], xc[s:s + win]
        if a.std() < min_rms or b.std() < min_rms:
            out.append(np.nan)
        else:
            out.append(gcc_phat_lag(a.astype(np.float64), b.astype(np.float64), max_lag))
    return np.array(out)


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and a.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    sep = NBSSSeparator(a.ckpt, device)

    files = sorted(glob.glob(os.path.join(a.scad_dir, "**", "*_mixture.wav"), recursive=True))
    assert files, f"keine *_mixture.wav unter {a.scad_dir}"
    if not a.all and a.n < len(files):
        files = list(np.random.RandomState(a.seed).choice(files, a.n, replace=False))
    os.makedirs(a.out, exist_ok=True)
    win = int(a.win * SR)

    rows = []
    for fi, mf in enumerate(files):
        cf = mf.replace("_mixture.wav", "_control.wav")
        mix, sr1 = sf.read(mf, dtype="float32", always_2d=True)
        ctl, sr2 = sf.read(cf, dtype="float32", always_2d=True)
        assert sr1 == sr2 == SR and mix.shape == ctl.shape and mix.shape[1] == 4, (mf, mix.shape, sr1)
        mix, ctl = mix.T, ctl.T
        speech_gt = mix - ctl
        T = mix.shape[1]

        speech = np.zeros_like(mix)
        bg = np.zeros_like(mix)
        for c in range(4):
            s, b = sep.separate(np.roll(mix, -c, axis=0))
            speech[c, :T], bg[c, :T] = s[:T], b[:T]

        for c in range(4):
            r = dict(file=os.path.basename(mf), ch=c,
                     sisdr_in=si_sdr(mix[c], speech_gt[c]),
                     sisdr_speech=si_sdr(speech[c], speech_gt[c]),
                     sisdr_bg=si_sdr(bg[c], ctl[c]))
            r["sisdri_speech"] = r["sisdr_speech"] - r["sisdr_in"]
            if c > 0:
                l_ctl = window_lags(ctl[0], ctl[c], win, a.max_lag)
                for name, sig in (("bg", bg), ("mix", mix)):
                    l = window_lags(sig[0], sig[c], win, a.max_lag)
                    ok = ~np.isnan(l) & ~np.isnan(l_ctl)
                    r[f"lag_agree_{name}"] = float((np.abs(l[ok] - l_ctl[ok]) <= 1).mean()) if ok.any() else np.nan
                    r[f"lag_mae_{name}"] = float(np.abs(l[ok] - l_ctl[ok]).mean()) if ok.any() else np.nan
            rows.append(r)
        print(f"[{fi + 1}/{len(files)}] {os.path.basename(mf)}")

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("file", "ch"), k))
    with open(os.path.join(a.out, "per_file.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Mittelwerte je Rollung (Kanal c am Referenzplatz) ===")
    cols = ["sisdr_in", "sisdr_speech", "sisdri_speech", "sisdr_bg", "lag_agree_bg", "lag_agree_mix", "lag_mae_bg"]
    print("ch  " + "  ".join(f"{k:>14}" for k in cols))
    for c in range(4):
        sel = [r for r in rows if r["ch"] == c]
        vals = []
        for k in cols:
            v = [r[k] for r in sel if k in r and not np.isnan(r[k])]
            vals.append(f"{np.mean(v):14.2f}" if v else f"{'-':>14}")
        print(f"{c:<3} " + "  ".join(vals))
    print(f"\nDetails: {os.path.join(a.out, 'per_file.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scad-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--win", type=float, default=1.0, help="Fensterlaenge GCC-PHAT in s")
    p.add_argument("--max-lag", type=int, default=8, help="max. Lag in Samples (Arraygroesse beachten)")
    p.add_argument("--gpu-frac", type=float, default=0.25)
    main(p.parse_args())