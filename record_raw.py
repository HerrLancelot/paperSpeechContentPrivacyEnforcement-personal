import argparse, os, time, json, datetime, threading, queue
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

p = argparse.ArgumentParser()
p.add_argument("--device", type=int, default=22)
p.add_argument("--minutes", type=float, default=5.0, help="Maximaldauer")
p.add_argument("--label", type=str, default="street")
p.add_argument("--sr", type=int, default=16000)
p.add_argument("--channels", type=int, default=6)
p.add_argument("--outdir", type=str, default="raw_recordings")
p.add_argument("--list", action="store_true")
a = p.parse_args()

if a.list:
    print(sd.query_devices()); raise SystemExit

os.makedirs(a.outdir, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base = os.path.join(a.outdir, f"{a.label}_{ts}")

chunks, markers = [], []
stop = threading.Event()
t0 = [None]
n_frames = [0]

def cb(indata, frames, time_info, status):
    if status:
        print("Status:", status)
    chunks.append(indata.copy())
    n_frames[0] += frames
    if n_frames[0] >= a.minutes * 60 * a.sr:
        stop.set()

def keys():
    while not stop.is_set():
        s = input().strip().lower()
        t = n_frames[0] / a.sr
        if s == "m":
            markers.append(round(t, 2)); print(f"  Marker bei {t:.1f} s")
        elif s == "q":
            stop.set()

print("Start in 5 s. 'm'+Enter = Sprecher gehört, 'q'+Enter = Stop & Speichern")
time.sleep(5)
threading.Thread(target=keys, daemon=True).start()
with sd.InputStream(samplerate=a.sr, channels=a.channels, dtype="int16",
                    device=a.device, callback=cb):
    print("Aufnahme läuft...")
    while not stop.is_set():
        time.sleep(0.1)

audio = np.concatenate(chunks, axis=0)
wavfile.write(base + ".wav", a.sr, audio)
json.dump({"label": a.label, "sr": a.sr, "channels": a.channels,
           "duration_s": len(audio) / a.sr, "speech_markers_s": markers},
          open(base + ".json", "w"), indent=2)
peak = np.abs(audio.astype(np.int32)).max()
print(f"Gespeichert: {base}.wav ({len(audio)/a.sr:.0f} s), Peak {peak}/32767"
      + ("  CLIPPING!" if peak >= 32760 else ""))