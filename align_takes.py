#!/usr/bin/env python3
"""
align_takes.py  --  Takes ueber die Sync-Chirps auf eine gemeinsame Zeitbasis bringen

align:  findet in jedem Take den Start- und Ende-Chirp des Hintergrund-Geraets (Aufwaertssweep) und den
        Start-Chirp des Sprecher-Geraets (Abwaertssweep), bestimmt die Positionen mit Sub-Sample-Genauigkeit
        (Matched Filter + Phasensteigung im Chirp-Band), schneidet den Inhaltsbereich aus und schreibt ihn als
        <take>_aligned.wav (alle Kanaele, gleiche Laenge, gleiche Zeitbasis wie die Datei).
        Optional --drift_correct: Zeitachse anhand von Start-/Ende-Chirp strecken (Clock-Drift Mikro vs. Wiedergabe).
snr:    gated SNR am Mikro aus ausgerichteten bg_only- und speech_only-Takes (optional Noise-Floor-Abzug).

Beispiele:
  python align_takes.py align --scene_dir stimuli/scene01 --takes recordings/scene01/*.wav --out_dir aligned --ref_ch 2
  python align_takes.py snr --bg aligned/scene01__bg_only__take01_aligned.wav \
        --speech aligned/scene01__speech_only__take01_aligned.wav --ch 2 [--noise aligned/..noise_floor..]

--ref_ch: Spalte der Aufnahme (0-basiert) fuer die Chirp-Suche, muss ein Roh-Mikrokanal sein.
  Bei 6-Kanal-Aufnahme (Index 0/1 sind KEINE Rohmikros): 2..5. Bei einer Aufnahme nur der genutzten Kanaele (4,3,2,5): 0.
Der Chirp wird an beliebiger Stelle der Aufnahme gefunden (Vorlauf egal), eine Szene pro Aufnahmedatei.
Die ermittelte Verschiebung wird auf ALLE Kanaele angewendet (gleiches Geraet, gleiche Clock).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, find_peaks, resample

from render_lab_stimuli import (CHIRP_F0, CHIRP_F1, T_BG_CHIRP, T_CONTENT, T_SP_CHIRP, gated_power, make_chirp)


def find_chirp(x, ref, lo, hi, fs, pad=256):
    """Sucht ref in x[lo:hi]. Rueckgabe: (fraktionale Startposition in x, Peak/Median-Verhaeltnis).
    Grob: Matched Filter (ganzzahlige Lage). Fein: Steigung der Kreuzspektrum-Phase im Chirp-Band
    (gewichtete Regression) -> Sub-Sample-Genauigkeit."""
    lo, hi = max(int(lo), 0), min(int(hi), len(x))
    if hi - lo < len(ref) + 2:
        raise ValueError("Chirp-Suchfenster liegt ausserhalb der Aufnahme (Aufnahme zu kurz oder erster Chirp falsch gefunden)")
    seg = x[lo:hi].astype(np.float64)
    c = fftconvolve(seg, ref[::-1].astype(np.float64), mode="valid")
    a = np.abs(c)
    i = int(np.argmax(a))
    quality = float(a[i] / (np.median(a) + 1e-20))

    start = lo + i - pad
    if start < 0 or start + len(ref) + 2 * pad > len(x):
        return float(lo + i), quality                       # zu nah am Rand: nur ganzzahlig
    win = x[start:start + len(ref) + 2 * pad].astype(np.float64)
    n = 1 << int(np.ceil(np.log2(len(win) + len(ref))))
    X, R = np.fft.rfft(win, n), np.fft.rfft(ref.astype(np.float64), n)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= CHIRP_F0) & (f <= CHIRP_F1)
    C = (X * np.conj(R))[band] * np.exp(2j * np.pi * f[band] * pad / fs)   # groeben Versatz (pad) entfernen
    w = np.abs(C)
    ph = np.unwrap(np.angle(C))
    # ph ~ -2*pi*f*d/fs + const  -> gewichtete lineare Regression ueber f
    fm, pm = np.average(f[band], weights=w), np.average(ph, weights=w)
    slope = np.sum(w * (f[band] - fm) * (ph - pm)) / np.sum(w * (f[band] - fm) ** 2)
    d = -slope * fs / (2 * np.pi)
    return float(start + pad + d), quality


def find_chirp_pair(xc, ref, fs, dt_end, tol_s=0.2):
    """Findet Start- und Ende-Chirp des Hintergrund-Geraets an beliebiger Stelle der Aufnahme:
    das staerkste Peak-Paar, dessen Abstand dt_end (s) +- tol_s betraegt. Rueckgabe: grobe Startindizes."""
    c = fftconvolve(xc.astype(np.float64), ref[::-1].astype(np.float64), mode="valid")
    a = np.abs(c)
    pk, _ = find_peaks(a, distance=int(0.25 * fs))
    pk = pk[np.argsort(a[pk])[::-1][:40]]
    best = None
    for i in pk:
        for j in pk:
            if abs((j - i) - dt_end * fs) <= tol_s * fs and (best is None or a[i] + a[j] > best[0]):
                best = (a[i] + a[j], int(i), int(j))
    if best is None:
        raise ValueError("kein passendes Chirp-Paar gefunden (Pegel/Kanal pruefen; Aufnahme komplett? Szene passt zum Manifest?)")
    return best[1], best[2]


def frac_advance(x, d):
    """y[n] = x[n + d] fuer fraktionales d (bandbegrenzt, per FFT-Phase). x: (T, C)."""
    n = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    X *= np.exp(2j * np.pi * np.fft.rfftfreq(n) * d)[:, None]
    return np.fft.irfft(X, n=n, axis=0)


def align_take(path, manifest, ref_ch, out_dir, drift_correct):
    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    tl = manifest["timeline"]
    D = tl["content_dur"]
    up, dn = make_chirp(fs, True), make_chirp(fs, False)
    xc = x[:, ref_ch]

    dt_end = tl["bg_chirp"][1] - T_BG_CHIRP
    i0c, i1c = find_chirp_pair(xc, up, fs, dt_end)
    w = int(0.05 * fs)
    p_bg0, q_bg0 = find_chirp(xc, up, i0c - w, i0c + w + len(up), fs)
    p_bg1, q_bg1 = find_chirp(xc, up, i1c - w, i1c + w + len(up), fs)
    exp_sp = p_bg0 + (T_SP_CHIRP - T_BG_CHIRP) * fs
    p_sp0, q_sp0 = find_chirp(xc, dn, exp_sp - 0.5 * fs, exp_sp + 0.5 * fs + len(dn), fs)

    ratio = (p_bg1 - p_bg0) / (dt_end * fs)          # 1.0 = keine Drift
    drift_ppm = (ratio - 1) * 1e6
    r = ratio if drift_correct else 1.0

    a = p_bg0 + (T_CONTENT - T_BG_CHIRP) * fs * r    # Inhaltsbeginn in Aufnahme-Samples (fraktional)
    n_out = int(round(D * fs))
    n_rec = int(round(D * fs * r))
    i0 = int(np.floor(a))
    if i0 < 0 or i0 + n_rec + 2 > len(x):
        raise RuntimeError(f"{path.name}: Inhaltsbereich liegt ausserhalb der Aufnahme")
    y = frac_advance(x, a - i0)
    seg = y[i0:i0 + n_rec]
    if n_rec != n_out:
        seg = resample(seg, n_out, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"{path.stem}_aligned.wav"
    sf.write(out_wav, seg.astype(np.float32), fs, subtype="FLOAT")
    sp_off_ms = ((p_sp0 - p_bg0) / fs - (T_SP_CHIRP - T_BG_CHIRP)) * 1000
    info = {
        "take": path.name, "fs": fs, "chirp_pos_samples": {"bg_start": p_bg0, "bg_end": p_bg1, "speech_start": p_sp0},
        "chirp_peak_quality": {"bg_start": q_bg0, "bg_end": q_bg1, "speech_start": q_sp0},
        "drift_ppm": drift_ppm, "drift_corrected": bool(drift_correct),
        "speech_device_offset_ms": sp_off_ms,
        "speech_start_in_aligned_s": manifest["speech"]["start_in_content_s"] + sp_off_ms / 1000,
        "aligned_file": out_wav.name,
    }
    (out_dir / f"{path.stem}_aligned.json").write_text(json.dumps(info, indent=2))
    warn = " [!] schwacher Chirp - Pegel/Kanal pruefen" if min(q_bg0, q_bg1, q_sp0) < 8 else ""
    if abs(drift_ppm) > 200:
        warn += " [!] Drift > 200 ppm: Aussetzer/Ueberlaeufe in der Aufnahme? Take pruefen"
    print(f"{path.name}: drift {drift_ppm:+.1f} ppm | Sprecher-Geraet {sp_off_ms:+.1f} ms | "
          f"Chirp-Qualitaet {q_bg0:.0f}/{q_bg1:.0f}/{q_sp0:.0f}{warn}")


def cmd_snr(args):
    def load(p):
        x, fs = sf.read(p, dtype="float64", always_2d=True)
        return x[:, args.ch], fs

    s, fs = load(args.speech)
    b, _ = load(args.bg)
    ps, pb = gated_power(s, fs), gated_power(b, fs)
    if args.noise:
        nz, _ = load(args.noise)
        pn = float(np.mean(nz ** 2))
        ps, pb = max(ps - pn, 1e-20), max(pb - pn, 1e-20)
        print(f"Noise-Floor-Leistung abgezogen: {10*np.log10(pn+1e-20):.1f} dB")
    print(f"gated Sprache {10*np.log10(ps):.1f} dB | gated Hintergrund {10*np.log10(pb):.1f} dB | "
          f"SNR am Mikro {10*np.log10(ps/pb):+.1f} dB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("align")
    a.add_argument("--scene_dir", required=True)
    a.add_argument("--takes", nargs="+", required=True)
    a.add_argument("--out_dir", default="aligned")
    a.add_argument("--ref_ch", type=int, default=2)
    a.add_argument("--drift_correct", action="store_true")
    s = sub.add_parser("snr")
    s.add_argument("--bg", required=True)
    s.add_argument("--speech", required=True)
    s.add_argument("--noise")
    s.add_argument("--ch", type=int, default=2)
    args = ap.parse_args()

    if args.cmd == "align":
        manifest = json.loads((Path(args.scene_dir) / "manifest.json").read_text())
        for t in args.takes:
            try:
                align_take(Path(t), manifest, args.ref_ch, Path(args.out_dir), args.drift_correct)
            except Exception as e:  # ein kaputter Take soll die anderen nicht stoppen
                print(f"{t}: FEHLER {e}")
    else:
        cmd_snr(args)


if __name__ == "__main__":
    main()