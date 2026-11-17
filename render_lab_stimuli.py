#!/usr/bin/env python3
"""
render_lab_stimuli.py  --  Stimuli fuer den Labor-Livetest (BA)

Rendert pro Szene VIER Dateien (je ein Satz fuer jedes Wiedergabegeraet):
  bg_full.wav      Hintergrund (ein Clip pro Lautsprecher-Slot) + Sync-Chirps   (n_out Kanaele, z. B. 8 fuer 7.1)
  bg_chirp.wav     nur die Sync-Chirps, sonst Stille (gleiche Laenge/Kanaele)
  speech_full.wav  LibriSpeech-Sprache + Sync-Chirps                           (1 Kanal)
  speech_chirp.wav nur die Sync-Chirps, sonst Stille
  manifest.json    Zeitleiste, Kanalbelegung, Pegel, Seed

Aufnahme-Varianten (siehe run_take.py) sind dann reine Dateiauswahl:
  combined     = bg_full  + speech_full
  bg_only      = bg_full  + speech_chirp
  speech_only  = bg_chirp + speech_full
  noise_floor  = bg_chirp + speech_chirp
Da die Chirps in ALLEN Varianten identisch vorhanden sind, lassen sich alle Takes
mit demselben Verfahren (align_takes.py) auf dieselbe Zeitbasis ausrichten.

Zeitleiste (Sekunden, relativ zum Dateianfang):
  1.0   Chirp (Aufwaertssweep) ueber EINEN Hintergrund-Lautsprecher  -> bg-Geraet
  2.5   Chirp (Abwaertssweep) ueber den Sprecher-Lautsprecher         -> Sprecher-Geraet
  4.0   Inhalt beginnt (Dauer --duration)
  ...   Ende-Chirps (gleiche Sweeps) zur Drift-Kontrolle

Beispiel:
  python render_lab_stimuli.py --out_dir stimuli/scene01 \
      --bg_files a.wav b.wav c.wav d.wav e.wav f.wav \
      --speech_files 84-121123-0000.flac 84-121123-0001.flac \
      --duration 30 --snr_db 6

  python render_lab_stimuli.py --out_dir stimuli --channel_test      # Kanal-Zuordnung pruefen

NEU (ein Geraet, z. B. HDMI-7.1, Sprache auf EINEM Lautsprecher):
  # Ambisonics-Hintergrund (FOA) wird auf --layout dekodiert (Winkel: --az, Standard ITU 7.1)
  python render_lab_stimuli.py --out_dir stimuli/amb01 --bg_foa soundscape_foa.wav --ambi_format acn_sn3d \
      --speech_files 84-121123-0000.flac --speech_slot FC --duration 30 --snr_db 6
  # Hintergrund aus Reaper/IEM (fertige 8-Kanal-Datei in Layout-Reihenfolge) statt eigenem Decoder:
  python render_lab_stimuli.py ... --bg_multich bg_8ch_from_reaper.wav
  # Einzelquellen (--bg_files/--bg_dir) lassen sich mit --bg_foa/--bg_multich kombinieren.
  # Ereignis-Clips nur einmal spielen (Rest Stille): --bg_files bed.wav event.wav --bg_slots FL,SR --bg_once SR
  Abspielen dann mit EINEM Geraet: run_take.py --bg_device X --speech_device X --speech_map <Geraetekanal des Sprechers>
  (bg_full + speech_full werden dort auf dem Geraetepuffer addiert; --speech_slot beruecksichtigt das beim Peak-Limit.)
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import chirp as sp_chirp
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Gemeinsame Konstanten (werden auch von run_take.py / align_takes.py benutzt)
# ---------------------------------------------------------------------------
CHIRP_DUR = 0.5
CHIRP_F0, CHIRP_F1 = 400.0, 6500.0   # < 8 kHz, damit er auch im 16-kHz-Mikrosignal liegt
CHIRP_PEAK_DB = -14.0
T_BG_CHIRP = 1.0
T_SP_CHIRP = 2.5
T_CONTENT = 4.0
END_GAP = 1.0          # Stille zwischen Inhaltsende und Ende-Chirp
END_SP_OFFSET = 1.5    # Ende-Chirp der Sprache liegt so viel nach dem Ende-Chirp des Hintergrunds
TAIL = 1.5

# WAV-Standardreihenfolge fuer 7.1; ANPASSEN, falls die Anlage anders belegt ist (--layout)
DEFAULT_LAYOUT = "FL,FR,FC,LFE,BL,BR,SL,SR"
DEFAULT_BG_SLOTS = "FL,FR,FC,BL,BR,SL,SR"  # LFE bleibt leer


def make_chirp(sr, up=True):
    """Logarithmischer Sweep mit 10-ms-Hann-Flanken. up=True: aufwaerts (Hintergrund), False: abwaerts (Sprecher)."""
    n = int(round(CHIRP_DUR * sr))
    t = np.arange(n) / sr
    f0, f1 = (CHIRP_F0, CHIRP_F1) if up else (CHIRP_F1, CHIRP_F0)
    x = sp_chirp(t, f0=f0, t1=CHIRP_DUR, f1=f1, method="logarithmic")
    nf = int(0.01 * sr)
    w = np.hanning(2 * nf)
    x[:nf] *= w[:nf]
    x[-nf:] *= w[nf:]
    return (x * 10 ** (CHIRP_PEAK_DB / 20)).astype(np.float32)


def timeline(duration):
    t_end_bg = T_CONTENT + duration + END_GAP
    t_end_sp = t_end_bg + END_SP_OFFSET
    total = t_end_sp + CHIRP_DUR + TAIL
    return {
        "bg_chirp": [T_BG_CHIRP, t_end_bg],
        "sp_chirp": [T_SP_CHIRP, t_end_sp],
        "content_start": T_CONTENT,
        "content_dur": float(duration),
        "total": total,
    }


def gated_power(x, sr, frame_s=0.020, thr_db=-40.0):
    """Mittlere Leistung ueber 'aktive' Frames (Frame-RMS > Max-Frame-RMS + thr_db).
    Entspricht audio_prep.active_rms (20-ms-Frames, Gate -40 dB unter dem lautesten Frame); Leistung statt RMS,
    das SNR-Verhaeltnis ist identisch (bis auf eps=1e-10 in active_rms)."""
    n = int(frame_s * sr)
    m = len(x) // n
    if m == 0:
        return float(np.mean(np.square(x, dtype=np.float64)))
    fr = x[: m * n].reshape(m, n).astype(np.float64)
    p = np.mean(fr ** 2, axis=1)
    pdb = 10 * np.log10(p + 1e-20)
    act = pdb > pdb.max() + thr_db
    return float(np.mean(p[act]))


def load_mono(path, sr):
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        fr = Fraction(sr, fs).limit_denominator(2000)
        x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return x


def tile_clip(clip, n, sr, rng, gap):
    """Wiederholt einen Clip bis n Samples. Zufaellige Startphase; gap=(lo,hi) Sekunden Stille zwischen Wiederholungen."""
    c = clip.copy()
    fade = min(int(0.01 * sr), len(c) // 4)
    if fade > 0:
        w = np.hanning(2 * fade)
        c[:fade] *= w[:fade]
        c[-fade:] *= w[fade:]
    out = np.zeros(n, np.float32)
    # Clip mindestens so lang wie die Szene (z. B. verkettete Hintergruende): Ausschnitt OHNE Wiederholungsnaht
    pos = -int(rng.uniform(0, len(c) - n + 1)) if len(c) >= n else -int(rng.uniform(0, len(c)))
    while pos < n:
        a, b = max(pos, 0), min(pos + len(c), n)
        out[a:b] = c[a - pos:b - pos]
        pos += len(c) + int(rng.uniform(gap[0], gap[1]) * sr)
    return out


def place_once(clip, n, sr, rng, margin_s=1.0):
    """Clip genau EINMAL an zufaelliger Stelle (>= margin_s Abstand zu den Raendern), sonst Stille.
    Rueckgabe: (Signal der Laenge n, Startzeit in s). Zu lange Clips werden gekuerzt."""
    m = int(margin_s * sr)
    c = clip[: max(n - 2 * m, 1)].copy()
    fade = min(int(0.01 * sr), len(c) // 4)
    if fade > 0:
        w = np.hanning(2 * fade)
        c[:fade] *= w[:fade]
        c[-fade:] *= w[fade:]
    a = int(rng.integers(m, max(n - len(c) - m, m) + 1))
    out = np.zeros(n, np.float32)
    out[a:a + len(c)] = c
    return out, a / sr


def edge_env(n, sr, fade_s=0.02):
    f = int(fade_s * sr)
    env = np.ones(n, np.float32)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    env[:f] = ramp
    env[-f:] = ramp[::-1]
    return env


def add_chirps(sig, times, chirp, sr, ch=None):
    for t in times:
        i = int(round(t * sr))
        if ch is None:
            sig[i:i + len(chirp)] += chirp
        else:
            sig[i:i + len(chirp), ch] += chirp


# ---------------------------------------------------------------------------
# Ambisonics (FOA) und vorgerenderte Mehrkanal-Hintergruende
# ---------------------------------------------------------------------------
# Azimut in Grad, 0 = vorne, + = links (gegen den Uhrzeigersinn); None = kein Lautsprecher (LFE)
AZ_DEFAULT = {"FL": 30.0, "FR": -30.0, "FC": 0.0, "LFE": None,
              "BL": 150.0, "BR": -150.0, "SL": 90.0, "SR": -90.0}


def parse_az(s, layout):
    az = dict(AZ_DEFAULT)
    if s:
        for item in s.split(","):
            k, v = item.split("=")
            az[k.strip()] = None if v.strip().lower() in ("none", "") else float(v)
    missing = [n for n in layout if n not in az]
    if missing:
        raise SystemExit(f"Kein Winkel fuer {missing}: --az \"{missing[0]}=<Grad>\" angeben (0 = vorne, + = links)")
    return az


def gated_power_multi(x, sr, frame_s=0.020, thr_db=-40.0):
    """Wie gated_power, aber fuer (T, C): Frame-Leistung = Summe ueber die Kanaele."""
    n = int(frame_s * sr)
    m = len(x) // n
    if m == 0:
        return float(np.sum(np.mean(np.square(x, dtype=np.float64), axis=0)))
    fr = x[: m * n].reshape(m, n, -1).astype(np.float64)
    p = np.sum(np.mean(fr ** 2, axis=1), axis=1)
    pdb = 10 * np.log10(p + 1e-20)
    return float(np.mean(p[pdb > pdb.max() + thr_db]))


def resample_to(x, fs, sr):
    if fs == sr:
        return x.astype(np.float32)
    fr = Fraction(sr, fs).limit_denominator(2000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)


def take_segment(x, n, rng, fade_s=0.1, sr=48000):
    """n Samples aus x (T, C): zufaelliger Ausschnitt; zu kurz -> mit Equal-Power-Crossfade wiederholen."""
    if len(x) >= n:
        o = int(rng.uniform(0, len(x) - n + 1))
        return x[o:o + n]
    f = min(int(fade_s * sr), len(x) // 4)
    r = np.linspace(0, 1, f, dtype=np.float32)[:, None]
    up, dn = np.sqrt(r), np.sqrt(1 - r)
    out = x.copy()
    while len(out) < n:
        out[-f:] = out[-f:] * dn + x[:f] * up
        out = np.concatenate([out, x[f:]], axis=0)
    return out[:n]


def load_foa(path, fmt, sr):
    """FOA-Datei -> (T, 4) in Reihenfolge W, X, Y, Z (SN3D)."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[1] < 4:
        raise SystemExit(f"{path}: {x.shape[1]} Kanaele, FOA braucht 4")
    x = x[:, :4]
    if fmt == "fuma":                      # Reihenfolge W X Y Z, W um 1/sqrt(2) abgesenkt
        W, X, Y, Z = x.T
        W = W * np.sqrt(2)
    else:                                  # ACN: W Y Z X
        W, Y, Z, X = x.T
        if fmt == "acn_n3d":               # N3D -> SN3D (1. Ordnung: 1/sqrt(3))
            Y, Z, X = (c / np.sqrt(3) for c in (Y, Z, X))
    return resample_to(np.stack([W, X, Y, Z], axis=1), fs, sr)


def rotate_yaw(wxyz, deg):
    if not deg:
        return wxyz
    p = np.deg2rad(deg)
    W, X, Y, Z = wxyz.T
    return np.stack([W, X * np.cos(p) - Y * np.sin(p), X * np.sin(p) + Y * np.cos(p), Z], axis=1)


def foa_decoder(az_list, weighting="maxre"):
    """Horizontaler FOA-Sampling-Decoder (W, X, Y; Z entfaellt, alle Lautsprecher auf Ohrhoehe).
    max-rE: 2D, 1. Ordnung, Gewicht cos(pi/4). Energie-normiert ueber alle Einfallsrichtungen.
    Rueckgabe: (n_spk, 3) fuer [W, X, Y] und die relative Energie je Richtung (5-Grad-Raster)."""
    th = np.deg2rad(np.asarray(az_list, float))
    a1 = np.cos(np.pi / 4) if weighting == "maxre" else 1.0
    D = np.stack([np.ones_like(th), 2 * a1 * np.cos(th), 2 * a1 * np.sin(th)], axis=1) / len(th)
    sw = np.deg2rad(np.arange(0, 360, 5.0))
    p = np.sum((D @ np.stack([np.ones_like(sw), np.cos(sw), np.sin(sw)])) ** 2, axis=0)
    return D / np.sqrt(p.mean()), p / p.mean()


def decode_foa(wxyz, layout, az, weighting):
    spk = [nm for nm in layout if az[nm] is not None]
    D, p = foa_decoder([az[nm] for nm in spk], weighting)
    print(f"  FOA-Decoder '{weighting}' auf {spk}: Energie ueber Einfallsrichtung +-{10*np.log10(p.max()/p.min())/2:.1f} dB")
    dec = wxyz[:, :3] @ D.T
    out = np.zeros((len(wxyz), len(layout)), np.float32)
    for j, nm in enumerate(spk):
        out[:, layout.index(nm)] = dec[:, j]
    return out


def pick_files(args, rng, n):
    if args.bg_files:
        return [Path(p) for p in args.bg_files]
    exts = {".wav", ".flac"}
    cand = sorted(p for p in Path(args.bg_dir).rglob("*") if p.suffix.lower() in exts)
    if len(cand) < n:
        raise SystemExit(f"--bg_dir enthaelt nur {len(cand)} Dateien, gebraucht: {n}")
    idx = rng.choice(len(cand), size=n, replace=False)
    return [cand[i] for i in idx]


def render_scene(args):
    sr, D = args.sr, args.duration
    rng = np.random.default_rng(args.seed)
    layout = [s.strip() for s in args.layout.split(",")]
    slots = [s.strip() for s in args.bg_slots.split(",")]
    for s in slots + [args.chirp_slot]:
        if s not in layout:
            raise SystemExit(f"Slot {s} nicht im Layout {layout}")
    n_out = len(layout)
    tl = timeline(D)
    total_n = int(round(tl["total"] * sr))
    c0, n = int(round(T_CONTENT * sr)), int(round(D * sr))
    env = edge_env(n, sr)

    # ---- Hintergrund ------------------------------------------------------
    if args.bg_files or args.bg_dir:
        n_src = args.n_bg if (args.bg_dir and not args.bg_files) else len(args.bg_files or [])
        n_src = min(n_src, len(slots))
        files = pick_files(args, rng, n_src)[:n_src]
    else:
        n_src, files = 0, []
    gap = tuple(args.bg_gap)
    src_db = args.bg_source_db
    bg_content = np.zeros((n, n_out), np.float32)
    bg_info = {}
    p_tot = 0.0
    once_slots = {t.strip() for t in args.bg_once.split(",") if t.strip()}
    for t in once_slots:
        if t not in slots:
            raise SystemExit(f"--bg_once {t} ist kein Hintergrund-Slot ({slots})")
    for slot, f in zip(slots, files):
        once_start = None
        if slot in once_slots:
            x, once_start = place_once(load_mono(f, sr), n, sr, rng)
        else:
            x = tile_clip(load_mono(f, sr), n, sr, rng, gap)
        p = gated_power(x, sr)
        x = x * np.sqrt(10 ** (src_db / 10) / max(p, 1e-20))
        p_tot += gated_power(x, sr)
        ch = layout.index(slot)
        bg_content[:, ch] = x * env
        bg_info[slot] = {"file": str(f), "out_channel_index": ch, "gated_rms_dbfs": src_db}
        if once_start is not None:
            bg_info[slot]["once_start_in_content_s"] = once_start
    # ---- Ambisonics / vorgerendertes Mehrkanal-File (optional, zusaetzlich zu Einzelquellen) --------------
    scene_db = args.bg_scene_db if args.bg_scene_db is not None else src_db
    layers = []
    if args.bg_foa:
        az = parse_az(args.az, layout)
        foa = take_segment(load_foa(args.bg_foa, args.ambi_format, sr), n, rng, sr=sr)
        layers.append(("foa", args.bg_foa, decode_foa(rotate_yaw(foa, args.ambi_rotate), layout, az, args.decoder),
                       {"format": args.ambi_format, "rotate_deg": args.ambi_rotate, "decoder": args.decoder,
                        "azimuth_deg": az}))
    if args.bg_multich:
        x, fs_m = sf.read(str(args.bg_multich), dtype="float32", always_2d=True)
        if x.shape[1] != n_out:
            raise SystemExit(f"--bg_multich hat {x.shape[1]} Kanaele, --layout hat {n_out} ({layout})")
        layers.append(("multich", args.bg_multich, take_segment(resample_to(x, fs_m, sr), n, rng, sr=sr),
                       {"note": "Kanalreihenfolge der Datei = --layout"}))
    for kind, f, layer, extra in layers:
        layer = layer * np.sqrt(10 ** (scene_db / 10) / max(gated_power_multi(layer, sr), 1e-20))
        p_tot += gated_power_multi(layer, sr)
        bg_content += layer * env[:, None]
        bg_info[kind] = {"file": str(f), "gated_total_dbfs": scene_db, **extra}
    if p_tot <= 0:
        raise SystemExit("Kein Hintergrund angegeben")
    bg_total_db = 10 * np.log10(max(p_tot, 1e-20))

    # ---- Sprache ------------------------------------------------------------
    sp = np.zeros(0, np.float32)
    gap_samples = np.zeros(int(args.speech_gap * sr), np.float32)
    s0 = int(round(args.speech_start * sr))
    utts, dropped = [], []
    for i, f in enumerate(args.speech_files):
        x = load_mono(f, sr)
        add = len(x) + (len(gap_samples) if len(sp) else 0)
        if s0 + len(sp) + add > n:      # passt nicht mehr komplett in --duration -> weglassen (lieber eine Aeusserung weniger)
            dropped = [str(g) for g in args.speech_files[i:]]
            break
        if len(sp):
            sp = np.concatenate([sp, gap_samples])
        utts.append({"file": str(f), "start_in_content_s": (s0 + len(sp)) / sr,
                     "end_in_content_s": (s0 + len(sp) + len(x)) / sr})
        sp = np.concatenate([sp, x])
    if dropped:
        print(f"[warn] {len(dropped)} Aeusserung(en) passen nicht mehr in --duration {D:.0f} s und werden NICHT eingebaut "
              f"(auch alle folgenden): {dropped}")
    if len(sp) == 0:
        raise SystemExit("Keine Aeusserung passt in --duration/--speech_start")
    sp = sp * edge_env(len(sp), sr, 0.02)
    target_sp_db = bg_total_db + args.snr_db
    sp_gain = np.sqrt(10 ** (target_sp_db / 10) / max(gated_power(sp, sr), 1e-20)) * 10 ** (args.speech_trim_db / 20)
    sp = sp * sp_gain

    # ---- Dateien zusammensetzen -------------------------------------------------
    up, dn = make_chirp(sr, True), make_chirp(sr, False)
    ch_chirp = layout.index(args.chirp_slot)

    bg_chirp = np.zeros((total_n, n_out), np.float32)
    add_chirps(bg_chirp, tl["bg_chirp"], up, sr, ch_chirp)
    bg_full = bg_chirp.copy()
    bg_full[c0:c0 + n] += bg_content

    sp_chirp_only = np.zeros(total_n, np.float32)
    add_chirps(sp_chirp_only, tl["sp_chirp"], dn, sr)
    sp_full = sp_chirp_only.copy()
    sp_full[c0 + s0:c0 + s0 + len(sp)] += sp

    peak = max(np.abs(bg_full).max(), np.abs(sp_full).max())
    if args.speech_slot:   # gleiches Geraet: run_take addiert die Sprache auf diesem Kanal zum Hintergrund
        if args.speech_slot not in layout:
            raise SystemExit(f"--speech_slot {args.speech_slot} nicht im Layout {layout}")
        peak = max(peak, float(np.abs(bg_full[:, layout.index(args.speech_slot)] + sp_full).max()))
    scale = 1.0
    if peak > 0.9:
        scale = 0.9 / peak
        print(f"[warn] Peak {peak:.2f} > 0.9 -> beide Dateien um {20*np.log10(scale):.1f} dB abgesenkt (SNR bleibt)")
        bg_full, bg_chirp, sp_full, sp_chirp_only = (a * scale for a in (bg_full, bg_chirp, sp_full, sp_chirp_only))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in [("bg_full", bg_full), ("bg_chirp", bg_chirp), ("speech_full", sp_full), ("speech_chirp", sp_chirp_only)]:
        sf.write(out / f"{name}.wav", arr.astype(np.float32), sr, subtype="PCM_24")

    manifest = {
        "sr": sr, "seed": args.seed, "timeline": tl, "layout": layout,
        "chirp": {"dur": CHIRP_DUR, "f0": CHIRP_F0, "f1": CHIRP_F1, "peak_db": CHIRP_PEAK_DB,
                  "bg": "up-sweep on slot " + args.chirp_slot, "speech": "down-sweep"},
        "bg_sources": bg_info, "bg_gap_s": list(gap), "bg_total_gated_db": float(bg_total_db),
        "speech": {"files": [str(f) for f in args.speech_files], "start_in_content_s": args.speech_start,
                   "dur_s": len(sp) / sr, "gap_s": args.speech_gap,
                   "utterances": utts, "dropped_files": dropped},
        "digital_snr_db_nominal": args.snr_db, "speech_trim_db": args.speech_trim_db, "global_scale": scale,
        "speech_slot": args.speech_slot,
        "gate": {"frame_s": 0.020, "thr_db": -40.0},
        "note": "Digitaler SNR != SNR am Mikro. Realen SNR aus bg_only/speech_only-Takes messen (align_takes.py snr).",
        "files": {k: f"{k}.wav" for k in ("bg_full", "bg_chirp", "speech_full", "speech_chirp")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Szene gerendert nach {out}  ({n_src} Hintergrundquellen, {tl['total']:.1f} s, bg gesamt {bg_total_db:.1f} dB, "
          f"Sprache {target_sp_db + args.speech_trim_db:.1f} dB gated)")


def render_channel_test(args):
    """Kanal k (0-basiert) piept (k+1)-mal. LFE bekommt einen 60-Hz-Ton. Sprecher-Datei: 1 Piepton."""
    sr = args.sr
    layout = [s.strip() for s in args.layout.split(",")]

    def beep(freq, dur=0.15):
        t = np.arange(int(dur * sr)) / sr
        x = 0.1 * np.sin(2 * np.pi * freq * t)
        f = int(0.01 * sr)
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
        return x.astype(np.float32)

    seg = int((0.3 + 0.4 * len(layout) + 1.0) * sr)
    sig = np.zeros((seg * len(layout), len(layout)), np.float32)
    for k, name in enumerate(layout):
        b = beep(60.0 if name == "LFE" else 1000.0)
        for j in range(k + 1):
            i = k * seg + int(0.3 * sr) + j * int(0.4 * sr)
            sig[i:i + len(b), k] = b
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / f"channel_test_{len(layout)}ch.wav", sig, sr, subtype="PCM_24")
    sp = np.zeros(int(2 * sr), np.float32)
    b = beep(1000.0)
    sp[int(0.3 * sr):int(0.3 * sr) + len(b)] = b
    sf.write(out / "channel_test_speech_1ch.wav", sp, sr, subtype="PCM_24")
    print(f"Kanaltest in {out}: Reihenfolge {layout} (Kanal k piept k+1 mal, je {seg/sr:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sr", type=int, default=48000, help="Renderrate (Interface-Rate, meist 48000)")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, help="Kanalnamen in Dateireihenfolge der bg-Datei")
    ap.add_argument("--channel_test", action="store_true", help="nur Kanaltest-Dateien erzeugen")
    ap.add_argument("--bg_files", nargs="*", help="ein Clip pro Slot (in Reihenfolge von --bg_slots)")
    ap.add_argument("--bg_dir", help="alternativ: Ordner, daraus werden --n_bg Clips zufaellig gezogen")
    ap.add_argument("--n_bg", type=int, default=6)
    ap.add_argument("--bg_slots", default=DEFAULT_BG_SLOTS, help="Layout-Namen, die Hintergrund bekommen")
    ap.add_argument("--chirp_slot", default="FC", help="Lautsprecher, ueber den der bg-Chirp laeuft")
    ap.add_argument("--bg_gap", type=float, nargs=2, default=[0.0, 0.0],
                    help="Stille (min max s) zwischen Clip-Wiederholungen; 0 0 = Endlosschleife; fuer Ereignisse z. B. 2 8")
    ap.add_argument("--bg_once", default="",
                    help="Slots (z. B. SL,BR), deren Clip nur EINMAL an zufaelliger Stelle gespielt wird (Ereignis-Clips)")
    ap.add_argument("--bg_source_db", type=float, default=-32.0, help="gated RMS (dBFS) pro Hintergrundquelle")
    ap.add_argument("--bg_foa", help="FOA-Ambisonics-Datei (4 Kanaele) als Hintergrund, wird auf --layout dekodiert")
    ap.add_argument("--ambi_format", default="acn_sn3d", choices=["acn_sn3d", "acn_n3d", "fuma"],
                    help="Konvention der FOA-Datei (README des Datensatzes pruefen)")
    ap.add_argument("--ambi_rotate", type=float, default=0.0, help="Szene um X Grad drehen (+ = links)")
    ap.add_argument("--decoder", default="maxre", choices=["maxre", "basic"])
    ap.add_argument("--az", default="", help='Lautsprecherwinkel ueberschreiben, z. B. "SL=100,SR=-100" (Grad, + = links)')
    ap.add_argument("--bg_multich", help="fertig gerenderte Mehrkanal-Datei (z. B. Reaper/IEM) in Layout-Reihenfolge")
    ap.add_argument("--bg_scene_db", type=float, default=None,
                    help="gated Gesamtpegel (dBFS, Summe aller Kanaele) von --bg_foa/--bg_multich; Standard = --bg_source_db")
    ap.add_argument("--speech_slot", help="Layout-Name des Lautsprechers, auf dem die Sprache bei EINEM Geraet liegt "
                                          "(nur fuer das Peak-Limit; das Mapping steht in run_take.py --speech_map)")
    ap.add_argument("--speech_files", nargs="+")
    ap.add_argument("--speech_gap", type=float, default=1.0, help="Pause zwischen Aeusserungen (s)")
    ap.add_argument("--speech_start", type=float, default=2.0, help="Sprachbeginn innerhalb des Inhalts (s)")
    ap.add_argument("--duration", type=float, default=30.0, help="Inhaltsdauer (s)")
    ap.add_argument("--snr_db", type=float, default=6.0, help="digitaler Ziel-SNR (gated, Summe der bg-Quellen)")
    ap.add_argument("--speech_trim_db", type=float, default=0.0, help="Zusatzpegel fuer Sprache (Lautsprecher-Abgleich)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.channel_test:
        render_channel_test(args)
        return
    if not args.speech_files:
        ap.error("--speech_files fehlt")
    if not (args.bg_files or args.bg_dir or args.bg_foa or args.bg_multich):
        ap.error("--bg_files, --bg_dir, --bg_foa oder --bg_multich noetig")
    render_scene(args)


if __name__ == "__main__":
    main()