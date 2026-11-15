#!/usr/bin/env python3
"""
make_lab_set.py - Szenen fuer den Labortest aus den Test-Manifesten zusammenstellen und rendern.

Zwei Schritte, weil die Daten auf dem Cluster liegen:

  1) select  (AUF DEM CLUSTER): zieht Eintraege aus den Manifesten (nur split=test), kopiert NUR diese Dateien
             nach <out>/lab_assets/ und schreibt <out>/lab_set.json (Pfade relativ, Referenztexte fuer den WER).
  2) render  (LOKAL, spaeter auch im Labor): rendert jede Szene mit render_lab_stimuli.py nach <out_dir>/<szene>/
             und legt dort refs.json ab (Fenster + Referenztext pro Aeusserung).

Manifest-Angabe ("Spec") - ein String, Teile mit ; getrennt, in Anfuehrungszeichen setzen:
    label=pfad/zur/manifest.csv;schluessel=wert;schluessel=wert ...
  Dateispalte wird erkannt: file | audio_filename | path. Relative Pfade gelten ab dem Ordner der Manifest-Datei,
  oder ab root=<ordner>. Ein Ordner statt einer CSV listet alle wav/flac darin auf.
  Ist eine Spalte `split` vorhanden, wird auf split=test gefiltert (ueberschreibbar: split=val).
  Jeder weitere Schluessel=Wert ist ein Spaltenfilter (z. B. kind=vehicle_event;exclude=False).
  Reservierte Schluessel (keine Filter):
    root=<ordner>      Basis fuer relative Pfade
    mode=loop|once|chain   loop: Clip wird in Schleife gespielt (Standard)
                           once: Clip wird genau EINMAL an zufaelliger Stelle gespielt (Ereignisse)
                           chain: aus mehreren Clips wird ein Hintergrund in voller Szenenlaenge gebaut
    group=spalte[,spalte]  chain: nur Clips mit gleichen Werten in diesen Spalten verketten
    order=spalte           chain: aufeinanderfolgende Werte (z. B. seg_idx) -> nahtlose Kette ohne Crossfade;
                           ohne order: zufaellige Clips der Gruppe mit 0,5-s-Crossfade
    w=<zahl>           Gewicht bei der zufaelligen Quellenwahl (Standard 1)

Beispiele (auf dem Cluster)
  python make_lab_set.py select --out lab_set --kinds foa,points --n_scenes 4 --duration 60 --seed 1 \
    --speech_manifest /pfad/librispeech_manifest_test-clean.csv \
    --foa_manifest /pfad/usotw_test.csv \
    --bg_manifest "demand=/pfad/demand_segments.csv;mode=chain;group=category,channel;order=seg_idx" \
    --bg_manifest "sonyc=/pfad/manifest.csv;mode=chain;group=sensor_id" \
    --bg_manifest "idmt=/pfad/idmt_manifest.csv;root=/pfad/projekt;kind=vehicle_event;exclude=False;mode=once"
  rsync -avP user@cluster:/pfad/lab_set/ ./lab_set/

Lokal
  python make_lab_set.py render --lab_set lab_set/lab_set.json --out_dir stimuli --az "SL=100,SR=-100"
  (alles Unbekannte geht an render_lab_stimuli.py: --az, --speech_trim_db, --bg_source_db, ...)

Kinds: foa    = ein USotW-FOA-Hintergrund (Ambisonics, auf --layout dekodiert)
       points = 1..3 Punktquellen (je ein Clip, je ein Lautsprecher-Slot ohne den Sprecher-Slot)
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SLOTS = ["FL", "FR", "FC", "BL", "BR", "SL", "SR"]      # wie DEFAULT_BG_SLOTS in render_lab_stimuli.py
SAFETY_S = 0.05                                          # Reserve gegen Rundungsfehler der Manifest-Dauern
FILE_COLS = ("file", "audio_filename", "path")
RESERVED = {"root", "mode", "group", "order", "w"}
AUDIO_EXT = {".wav", ".flac"}


# ------------------------------------------------------------------------------------------ Manifeste
def parse_spec(spec):
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    head = parts[0]
    label, path = head.split("=", 1) if "=" in head else (Path(head).stem, head)
    opts = {}
    for p in parts[1:]:
        if "=" not in p:
            raise SystemExit(f"Spec '{spec}': '{p}' ist kein schluessel=wert")
        k, v = p.split("=", 1)
        opts[k.strip()] = v.strip()
    return label.strip(), path.strip(), opts


def load_rows(spec, default_split="test"):
    """Liefert (label, Zeilen mit aufgeloestem Pfad in r['file'], opts)."""
    label, path, opts = parse_spec(spec)
    p = Path(path)
    if p.is_dir():
        rows = [{"file": str(f)} for f in sorted(p.rglob("*")) if f.suffix.lower() in AUDIO_EXT]
    else:
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit(f"{p}: leer")
        col = next((c for c in FILE_COLS if c in rows[0]), None)
        if col is None:
            raise SystemExit(f"{p}: keine Dateispalte ({'/'.join(FILE_COLS)}); Spalten: {list(rows[0])}")
        root = Path(opts["root"]) if "root" in opts else p.parent
        for r in rows:
            q = Path(r[col])
            r["file"] = str(q if q.is_absolute() else root / q)
    filters = {k: v for k, v in opts.items() if k not in RESERVED}
    if rows and "split" in rows[0] and "split" not in filters:
        filters["split"] = default_split
    for k in filters:
        if k not in rows[0]:
            raise SystemExit(f"{label}: Filterspalte '{k}' gibt es nicht (Spalten: {list(rows[0])})")
    n0 = len(rows)
    rows = [r for r in rows if all(str(r[k]).strip().lower() == v.lower() for k, v in filters.items())]
    if not rows:
        raise SystemExit(f"{label}: nach Filter {filters} bleibt nichts uebrig ({n0} Zeilen vorher)")
    if not Path(rows[0]["file"]).exists():
        raise SystemExit(f"{label}: Datei nicht gefunden: {rows[0]['file']} -> root=<ordner> angeben?")
    print(f"{label}: {len(rows)} von {n0} Zeilen nach Filter {filters or '-'}")
    return label, rows, opts


def duration_of(row):
    try:
        return float(row["duration_s"])
    except (KeyError, ValueError, TypeError):
        return float(sf.info(row["file"]).duration)


def local_name(path):
    """Eindeutiger Dateiname (DEMAND hat z. B. ueberall ch01.wav): letzte drei Pfadteile."""
    return "__".join(Path(path).parts[-3:])


def copy_asset(src, kind_dir, out):
    dst = out / "lab_assets" / kind_dir / local_name(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return str(dst.relative_to(out))


# ------------------------------------------------------------------------------------------ Hintergruende
def concat_xfade(arrs, sr, xf_s):
    out, f = arrs[0], int(xf_s * sr)
    for x in arrs[1:]:
        if f > 0 and len(out) >= f and len(x) >= f:
            r = np.linspace(0, 1, f, dtype=np.float32)
            out = np.concatenate([out[:-f], out[-f:] * np.sqrt(1 - r) + x[:f] * np.sqrt(r), x[f:]])
        else:
            out = np.concatenate([out, x])
    return out


def build_chain(label, rows, opts, rng, duration, out, name, dry_run):
    """Aus kurzen Clips einen Hintergrund der Szenenlaenge bauen. Rueckgabe: (relativer Pfad, Liste der Originale)."""
    group_cols = [c for c in opts.get("group", "").split(",") if c]
    order = opts.get("order")
    for c in group_cols + ([order] if order else []):
        if c not in rows[0]:
            raise SystemExit(f"{label}: Spalte '{c}' (group/order) gibt es nicht; Spalten: {list(rows[0])}")
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[c] for c in group_cols), []).append(r)
    clip_s = float(sf.info(rows[0]["file"]).duration)
    need = max(int(math.ceil((duration + 1.0) / clip_s)), 1)

    chosen, xfade = None, 0.5
    if order:
        runs = []
        for rs in groups.values():
            rs = sorted(rs, key=lambda r: int(r[order]))
            i = 0
            while i < len(rs):
                j = i
                while j + 1 < len(rs) and int(rs[j + 1][order]) == int(rs[j][order]) + 1:
                    j += 1
                if j - i + 1 >= need:
                    runs.append(rs[i:j + 1])
                i = j + 1
        if runs:
            run = runs[int(rng.integers(len(runs)))]
            o = int(rng.integers(0, len(run) - need + 1))
            chosen, xfade = run[o:o + need], 0.0
        else:
            print(f"[note] {label}: kein Lauf von {need} aufeinanderfolgenden Segmenten im Filter -> Crossfade-Kette")
    if chosen is None:
        pools = [rs for rs in groups.values() if len(rs) >= need] or list(groups.values())
        rs = pools[int(rng.integers(len(pools)))]
        chosen = [rs[int(i)] for i in rng.choice(len(rs), size=need, replace=len(rs) < need)]
    origs = [r["file"] for r in chosen]
    if dry_run:
        return None, origs
    arrs, sr = [], None
    for f in origs:
        x, fs = sf.read(f, dtype="float32", always_2d=True)
        if sr is not None and fs != sr:
            raise SystemExit(f"{label}: unterschiedliche Abtastraten in einer Kette ({f})")
        sr = fs
        arrs.append(x.mean(axis=1))
    y = concat_xfade(arrs, sr, xfade)
    dst = out / "lab_assets" / "points" / f"chain_{label}_{name}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, subtype="FLOAT")
    return str(dst.relative_to(out)), origs


# ------------------------------------------------------------------------------------------ select
def pick_speech(speech, rng, a):
    """Aeusserungen verschiedener Sprecher, bis --duration voll ist (alle passen komplett)."""
    t, chosen, used = a.speech_start, [], set()
    for i in rng.permutation(len(speech)):
        row, d = speech[i]
        spk = row.get("speaker_id")
        if spk is not None and spk in used:
            continue
        need = d + (a.speech_gap if chosen else 0)
        if t + need + SAFETY_S > a.duration:
            continue
        chosen.append(row)
        used.add(spk)
        t += need
        if a.duration - t < a.min_utt + a.speech_gap:
            break
    if not chosen:
        raise SystemExit("keine Aeusserung passt in --duration")
    return chosen


def cmd_select(a):
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    _, srows, _ = load_rows(a.speech_manifest)
    speech = [(r, duration_of(r)) for r in srows]
    speech = [(r, d) for r, d in speech if a.min_utt <= d <= a.max_utt]
    bg_sets = {}
    for spec in a.bg_manifest:
        label, rows, opts = load_rows(spec)
        bg_sets[label] = (rows, opts)
    foa = load_rows(a.foa_manifest)[1] if a.foa_manifest else []
    kinds = [k.strip() for k in a.kinds.split(",")]
    if "points" in kinds and not bg_sets:
        raise SystemExit("--kinds points braucht mindestens ein --bg_manifest")
    if "foa" in kinds and not foa:
        raise SystemExit("--kinds foa braucht --foa_manifest")
    free_slots = [s for s in SLOTS if s != a.speech_slot]
    labels = sorted(bg_sets)

    scenes, idx = [], 0
    for kind in kinds:
        for k in range(a.n_scenes):
            idx += 1
            name = f"{'foa' if kind == 'foa' else 'pts'}{k + 1:02d}"
            utts = pick_speech(speech, rng, a)
            scene = {"name": name, "kind": kind, "seed": a.seed * 1000 + idx, "duration": a.duration,
                     "snr_db": a.snr_db, "speech_start": a.speech_start, "speech_gap": a.speech_gap,
                     "speech_slot": a.speech_slot, "speech": []}
            for r in utts:
                scene["speech"].append({"file": None if a.dry_run else copy_asset(r["file"], "speech", out),
                                        "orig": r["file"], "speaker_id": r.get("speaker_id"),
                                        "text": r.get("text"), "duration_s": duration_of(r)})
            if kind == "foa":
                r = foa[int(rng.integers(len(foa)))]
                scene["bg"] = {"foa": None if a.dry_run else copy_asset(r["file"], "foa", out),
                               "orig": r["file"], "ambi_format": a.ambi_format}
                info = "Hintergrund foa"
            else:
                n_src = int(rng.integers(a.n_points[0], a.n_points[1] + 1))
                slots = [str(s) for s in rng.choice(free_slots, size=n_src, replace=False)]
                pts, seen = [], set()
                tries = 0
                while len(pts) < n_src:
                    tries += 1
                    if tries > 1000:
                        raise SystemExit("konnte keine verschiedenen Hintergrundquellen waehlen (zu wenig Dateien?)")
                    pool = labels
                    if not pts and not a.allow_event_only:     # mind. eine Dauer-Quelle (kein once), sonst fast nur Stille
                        pool = [l for l in labels if bg_sets[l][1].get("mode", "loop") != "once"] or labels
                    w = np.array([float(bg_sets[l][1].get("w", 1.0)) for l in pool])
                    label = str(rng.choice(pool, p=w / w.sum()))
                    rows, opts = bg_sets[label]
                    mode = opts.get("mode", "loop")
                    if mode == "chain":
                        file, orig = build_chain(label, rows, opts, rng, a.duration, out, f"{name}_{slots[len(pts)]}",
                                                 a.dry_run)
                        key = tuple(orig)
                    else:
                        r = rows[int(rng.integers(len(rows)))]
                        file = None if a.dry_run else copy_asset(r["file"], "points", out)
                        orig, key = r["file"], r["file"]
                    if key in seen:
                        continue
                    seen.add(key)
                    pts.append({"dataset": label, "slot": slots[len(pts)], "mode": mode, "orig": orig, "file": file})
                scene["bg"] = {"points": pts}
                info = "Hintergrund " + ", ".join(f"{p['slot']}={p['dataset']}({p['mode']})" for p in pts)
            scenes.append(scene)
            print(f"{name}: {len(utts)} Aeusserungen ({sum(s['duration_s'] for s in scene['speech']):.0f} s Sprache), {info}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "lab_set.json").write_text(json.dumps(
        {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "scenes": scenes}, indent=2, ensure_ascii=False))
    print(f"-> {out/'lab_set.json'}" + ("  (dry run: nichts kopiert)" if a.dry_run else f"  + {out/'lab_assets'}"))


# ------------------------------------------------------------------------------------------ render
def cmd_render(a, extra):
    js = Path(a.lab_set)
    base = js.parent
    ls = json.loads(js.read_text())
    script = Path(a.render_script) if a.render_script else Path(__file__).with_name("render_lab_stimuli.py")
    out_dir = Path(a.out_dir)
    for sc in ls["scenes"]:
        if a.only and sc["name"] not in a.only:
            continue
        files = [str(base / u["file"]) for u in sc["speech"]]
        cmd = [sys.executable, str(script), "--out_dir", str(out_dir / sc["name"]), "--duration", str(sc["duration"]),
               "--snr_db", str(sc["snr_db"]), "--seed", str(sc["seed"]), "--speech_files", *files,
               "--speech_gap", str(sc["speech_gap"]), "--speech_start", str(sc["speech_start"]),
               "--speech_slot", sc["speech_slot"]]
        bg = sc["bg"]
        if "foa" in bg:
            cmd += ["--bg_foa", str(base / bg["foa"]), "--ambi_format", bg["ambi_format"]]
        else:
            cmd += ["--bg_files", *[str(base / p["file"]) for p in bg["points"]],
                    "--bg_slots", ",".join(p["slot"] for p in bg["points"])]
            once = [p["slot"] for p in bg["points"] if p.get("mode") == "once"]
            if once:
                cmd += ["--bg_once", ",".join(once)]
        cmd += extra
        print(f"\n== {sc['name']} ==")
        subprocess.run(cmd, check=True)
        man = json.loads((out_dir / sc["name"] / "manifest.json").read_text())
        by_file = {str(base / u["file"]): u for u in sc["speech"]}
        refs = [{**by_file[u["file"]], "start_in_content_s": u["start_in_content_s"],
                 "end_in_content_s": u["end_in_content_s"]} for u in man["speech"]["utterances"]]
        (out_dir / sc["name"] / "refs.json").write_text(json.dumps(
            {"scene": sc["name"], "bg": bg, "bg_render": man.get("bg_sources"), "utterances": refs,
             "dropped": man["speech"].get("dropped_files", [])}, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="auf dem Cluster: Eintraege waehlen und Dateien kopieren")
    s.add_argument("--out", default="lab_set")
    s.add_argument("--speech_manifest", required=True)
    s.add_argument("--bg_manifest", action="append", default=[], help="Spec, mehrfach (siehe oben)")
    s.add_argument("--foa_manifest", help="Spec oder Ordner mit den FOA-Dateien")
    s.add_argument("--kinds", default="foa,points")
    s.add_argument("--n_scenes", type=int, default=4, help="Szenen pro Kind")
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--speech_start", type=float, default=2.0)
    s.add_argument("--speech_gap", type=float, default=3.0)
    s.add_argument("--min_utt", type=float, default=3.0)
    s.add_argument("--max_utt", type=float, default=15.0)
    s.add_argument("--snr_db", type=float, default=6.0)
    s.add_argument("--speech_slot", default="FC")
    s.add_argument("--n_points", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"))
    s.add_argument("--ambi_format", default="acn_sn3d", choices=["acn_sn3d", "acn_n3d", "fuma"])
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--allow_event_only", action="store_true",
                   help="Szenen nur aus once-Ereignissen erlauben (Standard: mind. eine Dauer-Quelle pro Szene)")
    s.add_argument("--dry_run", action="store_true", help="nur auswaehlen, nichts kopieren oder bauen")
    r = sub.add_parser("render", help="lokal: Szenen aus lab_set.json rendern")
    r.add_argument("--lab_set", default="lab_set/lab_set.json")
    r.add_argument("--out_dir", default="stimuli")
    r.add_argument("--render_script")
    r.add_argument("--only", nargs="*", help="nur diese Szenen (z. B. foa01 pts02)")
    a, extra = ap.parse_known_args()
    if a.cmd == "select":
        if extra:
            ap.error(f"unbekannte Argumente: {extra}")
        cmd_select(a)
    else:
        cmd_render(a, extra)


if __name__ == "__main__":
    main()