#!/usr/bin/env python3
"""
run_take.py  --  einen Take abspielen und gleichzeitig mit dem ReSpeaker aufnehmen

Spielt die von render_lab_stimuli.py erzeugten Dateien ueber ein oder zwei Audiogeraete ab
und nimmt dabei die Mikro-Rohkanaele auf. Die Synchronisation passiert NICHT hier, sondern
nachtraeglich ueber die Chirps (align_takes.py). Hier geht es nur darum, dass Aufnahme
etwas VOR der Wiedergabe startet und etwas NACH ihrem Ende stoppt.

Varianten:
  combined | bg_only | speech_only | noise_floor     (siehe render_lab_stimuli.py)

Beispiele:
  python run_take.py --list-devices
  python run_take.py --scene_dir stimuli/scene01 --variant combined \
      --bg_device "HDMI" --bg_map 1,2,3,4,5,6,7,8 \
      --speech_device "Lautsprecher" --speech_map 1 \
      --mic_device 20 --mic_channels 6 --mic_sr 16000

  # nur abspielen (live_anonymize.py nimmt das Mikro und die Rohkanaele auf; Wiedergabe wird protokolliert):
  python run_take.py --scene_dir stimuli/scene01 --variant combined ... --no_record

Gleiches Geraet fuer bg und Sprache (Interface mit >= 9 Ausgaengen): --bg_device und
--speech_device identisch angeben, --speech_map z. B. 9 -> es wird EIN Stream geoeffnet.

Hinweis: Mikro (WDM-KS) ist wahrscheinlich exklusiv -> nicht gleichzeitig mit live_anonymize.py aufnehmen.
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

VARIANTS = {
    "combined":    ("bg_full",  "speech_full"),
    "bg_only":     ("bg_full",  "speech_chirp"),
    "speech_only": ("bg_chirp", "speech_full"),
    "noise_floor": ("bg_chirp", "speech_chirp"),
}


def dev_arg(s):
    if s is None:
        return None
    return int(s) if str(s).isdigit() else s


def parse_map(s, n):
    """1-basierte Geraetekanaele; Eintrag i = Ausgangskanal fuer Dateikanal i."""
    m = list(range(1, n + 1)) if not s else [int(v) for v in s.split(",")]
    if len(m) != n:
        raise SystemExit(f"Kanal-Mapping hat {len(m)} Eintraege, Datei hat {n} Kanaele")
    return m


def build_device_buffers(sinks, total):
    """sinks: {device: [(data[T,C], mapping), ...]} -> {device: buffer[T, n_dev_ch]} (Quellen auf demselben Geraet werden gemischt)."""
    buffers = {}
    for dev, items in sinks.items():
        n_ch = max(max(m) for _, m in items)
        buf = np.zeros((total, n_ch), np.float32)
        for data, mapping in items:
            for i, m in enumerate(mapping):
                buf[:len(data), m - 1] += data[:, i]
        buffers[dev] = buf
    return buffers


def make_out_stream(dev, buf, sr, done):
    state = {"pos": 0}

    def cb(outdata, frames, t, status):
        if status:
            print(f"[out {dev}] {status}", file=sys.stderr)
        a = state["pos"]
        chunk = buf[a:a + frames]
        outdata[:len(chunk)] = chunk
        if len(chunk) < frames:
            outdata[len(chunk):] = 0
            raise sd.CallbackStop
        state["pos"] = a + frames

    return sd.OutputStream(samplerate=sr, device=dev, channels=buf.shape[1], dtype="float32",
                           latency="high", callback=cb, finished_callback=done.set)


def next_take_path(rec_dir, scene, variant):
    rec_dir.mkdir(parents=True, exist_ok=True)
    k = len(list(rec_dir.glob(f"{scene}__{variant}__take*.wav"))) + 1
    return rec_dir / f"{scene}__{variant}__take{k:02d}.wav"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--scene_dir")
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--bg_device", help="Index oder Namensteil")
    ap.add_argument("--bg_map", help="1-basierte Geraetekanaele je Dateikanal, z. B. 1,2,3,4,5,6,7,8")
    ap.add_argument("--speech_device")
    ap.add_argument("--speech_map", default="1", help="Geraetekanal fuer den Sprecher (1-basiert)")
    ap.add_argument("--mic_device")
    ap.add_argument("--mic_channels", type=int, default=6)
    ap.add_argument("--mic_sr", type=int, default=16000)
    ap.add_argument("--rec_dir", default="recordings")
    ap.add_argument("--preroll", type=float, default=1.5, help="Aufnahme laeuft so lange vor der Wiedergabe (s)")
    ap.add_argument("--postroll", type=float, default=1.5)
    ap.add_argument("--countdown", type=int, default=3, help="Sekunden Countdown vor dem Start (zum Positionieren)")
    ap.add_argument("--no_record", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        print("\nHost-APIs:", [h["name"] for h in sd.query_hostapis()])
        return
    for req in ("scene_dir", "variant", "bg_device", "speech_device"):
        if getattr(args, req) is None:
            ap.error(f"--{req} fehlt")
    if not args.no_record and args.mic_device is None:
        ap.error("--mic_device fehlt (oder --no_record)")

    scene_dir = Path(args.scene_dir)
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    bg_name, sp_name = VARIANTS[args.variant]
    bg, sr_bg = sf.read(scene_dir / f"{bg_name}.wav", dtype="float32", always_2d=True)
    sp, sr_sp = sf.read(scene_dir / f"{sp_name}.wav", dtype="float32", always_2d=True)
    if sr_bg != sr_sp:
        raise SystemExit("Abtastraten der Dateien unterscheiden sich")
    sr = sr_bg
    total = max(len(bg), len(sp))

    sinks = {}
    sinks.setdefault(dev_arg(args.bg_device), []).append((bg, parse_map(args.bg_map, bg.shape[1])))
    sinks.setdefault(dev_arg(args.speech_device), []).append((sp, parse_map(args.speech_map, sp.shape[1])))
    buffers = build_device_buffers(sinks, total)
    for dev, buf in buffers.items():
        pk = float(np.abs(buf).max())
        print(f"  Geraet {dev!r}: {buf.shape[1]} Ausgangskanaele, Spitze {20 * np.log10(pk + 1e-12):.1f} dBFS")
        if pk > 1.0:
            raise SystemExit(f"Clipping im Geraetepuffer {dev!r} (Spitze {pk:.2f}): bg und Sprache addieren sich auf "
                             f"demselben Kanal -> render_lab_stimuli.py mit --speech_slot neu rendern")

    dones = {dev: threading.Event() for dev in buffers}
    out_streams = [make_out_stream(dev, buf, sr, dones[dev]) for dev, buf in buffers.items()]

    scene = scene_dir.name
    rec_path = None
    mic = None
    stop_writer = threading.Event()
    q = queue.Queue()
    mic_status = []
    if not args.no_record:
        rec_path = next_take_path(Path(args.rec_dir) / scene, scene, args.variant)

        def in_cb(indata, frames, t, status):
            if status:
                mic_status.append(str(status))
            q.put(indata.copy())

        mic = sd.InputStream(samplerate=args.mic_sr, device=dev_arg(args.mic_device), channels=args.mic_channels,
                             dtype="float32", callback=in_cb)

        def writer():
            with sf.SoundFile(str(rec_path), "w", samplerate=args.mic_sr, channels=args.mic_channels,
                              subtype="FLOAT") as f:
                while not stop_writer.is_set() or not q.empty():
                    try:
                        f.write(q.get(timeout=0.2))
                    except queue.Empty:
                        pass

        wt = threading.Thread(target=writer, daemon=True)

    print(f"Szene {scene} | Variante {args.variant} | {total/sr:.1f} s | Geraete: {list(buffers)}")
    for i in range(args.countdown, 0, -1):
        print(f"  Start in {i} ...", flush=True)
        time.sleep(1)

    t_mic = t_play0 = t_play1 = None
    try:
        if mic is not None:
            wt.start()
            mic.start()
            t_mic = time.perf_counter()
            time.sleep(args.preroll)
        t_play0 = time.perf_counter()
        for s in out_streams:
            s.start()
        t_play1 = time.perf_counter()
        for dev, e in dones.items():
            e.wait(timeout=total / sr + 30)
        time.sleep(args.postroll)
    except KeyboardInterrupt:
        print("Abbruch -> Aufnahme wird gespeichert")
    finally:
        for s in out_streams:
            s.stop()
            s.close()
        if mic is not None:
            mic.stop()
            mic.close()
            stop_writer.set()
            wt.join()

    if args.no_record:
        log = Path(args.rec_dir) / scene / "playback_log.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps({"wallclock": time.strftime("%Y-%m-%d %H:%M:%S"), "scene": scene, "variant": args.variant,
                                "files": [bg_name, sp_name], "duration_s": total / sr,
                                "start_skew_between_output_streams_s": (t_play1 - t_play0) if t_play1 else None,
                                "argv": sys.argv}) + "\n")
        print(f"Wiedergabe protokolliert: {log}  (Zeitstempel zum Zuordnen der live_anonymize-Rohaufnahme)")
    if rec_path is not None:
        meta = {
            "scene_dir": str(scene_dir), "variant": args.variant, "files": [bg_name, sp_name],
            "playback_sr": sr, "mic_sr": args.mic_sr, "mic_channels": args.mic_channels,
            "devices": {str(d): sd.query_devices(d)["name"] for d in list(buffers) + [dev_arg(args.mic_device)]
                        if d is not None},
            "start_skew_between_output_streams_s": (t_play1 - t_play0) if t_play1 else None,
            "preroll_s": args.preroll, "postroll_s": args.postroll,
            "mic_status": mic_status, "manifest_seed": manifest.get("seed"),
            "argv": sys.argv,
        }
        rec_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"Aufnahme: {rec_path}" + (f"  [Mikro-Warnungen: {len(mic_status)}]" if mic_status else ""))


if __name__ == "__main__":
    main()