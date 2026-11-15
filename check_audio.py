#!/usr/bin/env python3
"""
check_audio.py - schneller Check, was die Workstation fuer 7.1 kann (Linux, geht aber auch sonst).

  python check_audio.py info
      Geraete mit Kanalzahl, unterstuetzte Sampleraten, ALSA-Karten, HDMI-ELD (was meldet der
      Receiver?), Audio-Server (PipeWire/PulseAudio), Gruppenrechte.

  python check_audio.py test --device "hw:0,3" --channels 8 --fs 48000
      Spielt Rauschen nacheinander auf jedem Kanal. Du hoerst, welcher Lautsprecher dran ist,
      und notierst die Reihenfolge -> daraus --order fuer render_7p1.py.

  python check_audio.py test --device "hw:0,3" --mic "ReSpeaker" --mic_ch 2,3,4,5
      Wie oben, nimmt dabei den ReSpeaker auf und zeigt den Pegel pro Mikro und Lautsprecher.
      Zeigt auch, ob Aufnahme und Wiedergabe gleichzeitig laufen (unter Windows/WDM-KS ging das nicht).

Abhaengigkeiten: pip install sounddevice numpy   (Linux: sudo apt install libportaudio2 alsa-utils)
"""
import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

TEST_RATES = (48000, 96000, 44100, 16000)


def sd_or_exit():
    try:
        import sounddevice as sd
        return sd
    except Exception as e:  # PortAudio fehlt oder Paket fehlt
        print(f"sounddevice nicht nutzbar: {e}\n"
              "  pip install sounddevice   und unter Linux:  sudo apt install libportaudio2")
        sys.exit(1)


def run(cmd, timeout=6):
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"(Fehler: {e})"


def parse_dev(s):
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return s


# --------------------------------------------------------------------------- info
def eld_summary():
    """HDMI-ELD: was der angeschlossene Receiver/Monitor an Audio meldet (nur Linux)."""
    keys = ("monitor_name", "connection_type", "eld_valid", "speakers",
            "sad", "support_ai")
    found = False
    for f in sorted(glob.glob("/proc/asound/card*/eld#*")):
        try:
            txt = open(f).read()
        except OSError:
            continue
        if "monitor_name" not in txt or "monitor_name\t\t\n" in txt:
            continue
        lines = [l for l in txt.splitlines() if l.startswith(keys)]
        if not any(l.startswith("monitor_name") and l.split(None, 1)[1:] for l in lines):
            continue
        found = True
        print(f"  {f}")
        for l in lines:
            if l.startswith("sad") and not any(k in l for k in ("coding_type", "channels", "rates")):
                continue
            print("    " + " ".join(l.split()))
    if not found:
        print("  keine gueltige ELD gefunden: Receiver nicht erkannt oder HDMI-Audio inaktiv "
              "(Receiver an, richtiger Eingang?)")


def cmd_info(_):
    sd = sd_or_exit()
    print(f"System: {platform.platform()}, Python {platform.python_version()}, "
          f"sounddevice {sd.__version__}, PortAudio {sd.get_portaudio_version()[1]}")
    print("\n== Geraete (sounddevice) ==")
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    for i, d in enumerate(devs):
        flag = "  <== >= 8 Ausgaenge" if d["max_output_channels"] >= 8 else ""
        print(f"[{i:2d}] {d['name']}  ({apis[d['hostapi']]['name']})  "
              f"out={d['max_output_channels']} in={d['max_input_channels']} "
              f"default_fs={d['default_samplerate']:.0f}{flag}")
    print("\n== Sampleraten fuer 8-Kanal-Ausgabe ==")
    for i, d in enumerate(devs):
        if d["max_output_channels"] < 8:
            continue
        ok = []
        for r in TEST_RATES:
            try:
                sd.check_output_settings(device=i, channels=8, samplerate=r, dtype="float32")
                ok.append(r)
            except Exception:
                pass
        print(f"[{i:2d}] {d['name']}: 8 Kanaele bei {ok if ok else 'keiner getesteten Rate'}")
    if platform.system() == "Linux":
        print("\n== ALSA-Karten (aplay -l) ==")
        print(run(["aplay", "-l"]) or "aplay nicht gefunden (sudo apt install alsa-utils)")
        print("\n== HDMI-ELD (was der Receiver meldet) ==")
        eld_summary()
        print("\n== Audio-Server ==")
        pa = run(["pactl", "info"])
        if pa:
            print("\n".join(l for l in pa.splitlines()
                            if l.startswith(("Server Name", "Server Version", "Default Sink"))))
            print("Karten-Profile mit 7.1: pactl list cards | grep -i -E 'surround-7.1|Name:'")
        else:
            print("  pactl nicht gefunden (kein PulseAudio/PipeWire-Pulse oder nicht installiert)")
        wp = run(["wpctl", "status"])
        if wp:
            print("  wpctl status (Anfang):\n    " + "\n    ".join(wp.splitlines()[:12]))
        grp = run(["id", "-nG"]) or ""
        print(f"\nGruppen: {grp}  -> 'audio' enthalten: {'audio' in grp.split()}")
        busy = run(["fuser", "-v"] + glob.glob("/dev/snd/pcm*")) if glob.glob("/dev/snd/pcm*") else None
        if busy:
            print("\nProzesse, die ALSA-Geraete belegen:\n" + busy)
    print("\nNaechster Schritt: python check_audio.py test --device <Name oder Index> --channels 8")


# --------------------------------------------------------------------------- test
def pink_burst(fs, dur, level_db):
    n = int(fs * dur)
    rng = np.random.default_rng(1)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(X, n)
    x /= np.sqrt(np.mean(x ** 2))
    x *= 10 ** (level_db / 20)
    w = int(0.02 * fs)
    x[:w] *= np.linspace(0, 1, w)
    x[-w:] *= np.linspace(1, 0, w)
    return x.astype(np.float32)


def cmd_test(a):
    sd = sd_or_exit()
    dev = parse_dev(a.device)
    names = a.names.split(",") if a.names else [f"ch{i}" for i in range(a.channels)]
    if len(names) != a.channels:
        sys.exit(f"--names hat {len(names)} Eintraege, --channels ist {a.channels}")
    try:
        sd.check_output_settings(device=dev, channels=a.channels, samplerate=a.fs, dtype="float32")
    except Exception as e:
        sys.exit(f"Geraet {dev!r} kann {a.channels} Kanaele bei {a.fs} Hz nicht: {e}\n"
                 "-> 'info' ausfuehren, anderes Geraet/andere Rate/anderes Audio-Profil waehlen")

    burst = pink_burst(a.fs, a.seconds, a.level_db)
    gap = np.zeros(int(0.6 * a.fs), np.float32)
    seg = len(burst) + len(gap)
    sig = np.zeros((seg * a.channels + int(0.5 * a.fs), a.channels), np.float32)
    for c in range(a.channels):
        sig[c * seg: c * seg + len(burst), c] = burst

    chunks, t_play = [], {}
    stream = None
    if a.mic is not None:
        mic_dev = parse_dev(a.mic)
        if isinstance(mic_dev, str) and not mic_dev.isdigit():
            hits = [i for i, d in enumerate(sd.query_devices())
                    if mic_dev.lower() in d["name"].lower() and d["max_input_channels"] > 0]
            if not hits:
                sys.exit(f"Kein Eingabegeraet mit '{mic_dev}' im Namen gefunden (siehe 'info')")
            mic_dev = hits[0]
        n_in = sd.query_devices(mic_dev)["max_input_channels"]
        print(f"Mikro: [{mic_dev}] {sd.query_devices(mic_dev)['name']}, {n_in} Kanaele, {a.mic_fs} Hz")
        stream = sd.InputStream(device=mic_dev, channels=n_in, samplerate=a.mic_fs, dtype="float32",
                                callback=lambda x, f, t, s: chunks.append(x.copy()))
        stream.start()
        time.sleep(0.7)
        t_play["n0"] = sum(len(c) for c in chunks)

    print(f"Spiele {a.channels} Kanaele nacheinander ({a.seconds:.1f} s je Kanal) auf {dev!r} ...")
    for c in range(a.channels):
        print(f"  Kanal {c} ({names[c]}) in {c * seg / a.fs:4.1f} s")
    sd.play(sig, a.fs, device=dev)
    sd.wait()
    print("Fertig. Notiere pro Index, welcher Lautsprecher gespielt hat "
          "(Position in --order von render_7p1.py = Kanalindex).")

    if stream is not None:
        time.sleep(0.4)
        stream.stop()
        stream.close()
        rec = np.concatenate(chunks, axis=0)
        mic_ch = [int(x) for x in a.mic_ch.split(",")] if a.mic_ch else list(range(rec.shape[1]))
        mic_ch = [m for m in mic_ch if m < rec.shape[1]]
        print(f"\nPegel am Mikro in dBFS (Zeile = Wiedergabekanal, Spalte = Mikrokanal {mic_ch}):")
        print("  " + "kanal".ljust(10) + "".join(f"mic{m}".rjust(8) for m in mic_ch))
        floor = rec[: t_play["n0"] - int(0.1 * a.mic_fs)][:, mic_ch]
        floor_db = 20 * np.log10(np.sqrt(np.mean(floor ** 2, axis=0)) + 1e-9)
        print("  " + "Stille".ljust(10) + "".join(f"{v:8.1f}" for v in floor_db))
        for c in range(a.channels):
            s0 = t_play["n0"] + int((c * seg / a.fs + 0.25) * a.mic_fs)
            s1 = t_play["n0"] + int((c * seg / a.fs + a.seconds) * a.mic_fs)
            x = rec[s0:s1][:, mic_ch]
            db = 20 * np.log10(np.sqrt(np.mean(x ** 2, axis=0)) + 1e-9)
            print("  " + f"{c} {names[c]}".ljust(10) + "".join(f"{v:8.1f}" for v in db))
        print("Zeile deutlich ueber 'Stille' = Lautsprecher kommt am Mikro an. Alles gleich laut "
              "oder nahe 'Stille' -> falsches Geraet/Profil oder Receiver-Eingang.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="Geraete, Kanaele, ELD, Audio-Server").set_defaults(fn=cmd_info)
    t = sub.add_parser("test", help="Kanaltest (optional mit Mikro-Pegel)")
    t.add_argument("--device", help='Index oder Name, z.B. "hw:0,3" oder 12 (siehe info)')
    t.add_argument("--channels", type=int, default=8)
    t.add_argument("--fs", type=int, default=48000)
    t.add_argument("--seconds", type=float, default=1.5)
    t.add_argument("--level_db", type=float, default=-35.0, help="Pegel des Rauschens in dBFS (leise starten!)")
    t.add_argument("--names", help="Anzeigenamen, z.B. FL,FR,FC,LFE,BL,BR,SL,SR")
    t.add_argument("--mic", help="Eingabegeraet (Index oder Namensteil, z.B. ReSpeaker)")
    t.add_argument("--mic_fs", type=int, default=16000)
    t.add_argument("--mic_ch", help="auszuwertende Mikrokanaele, z.B. 2,3,4,5")
    t.set_defaults(fn=cmd_test)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()