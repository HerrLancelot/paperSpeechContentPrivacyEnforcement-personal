"""Live-Mikrofon (ReSpeaker, 4 Kanaele, 16 kHz) -> Weg A -> anonymisierte wav, mit Messung.

Das Mikrofon-Callback legt nur Chunks in eine Queue. Der Hauptthread schiebt sie in den StreamAnonymizer
(Ringpuffer, Block + Kontext). Geschrieben wird ausschliesslich das anonymisierte Ergebnis; Rohaudio bleibt im
Speicher des Puffers und wird nie auf die Platte geschrieben. Reste kleiner als ein Block werden verworfen.

Gemessen wird (Kriterium: Verarbeitung pro Block < Blockdauer, Puffer laeuft nicht ueber):
  RTF je Block, Backlog (Sekunden Audio, die in der Queue auf Verarbeitung warten), Latenz ab letztem
  Sample des Blocks, Eingabe-Ueberlaeufe der Soundkarte (status-Meldungen von PortAudio).

  python live_anonymize.py --list-devices
  python live_anonymize.py --ckpt nbss/ckpts/epoch70_metric23.9507.ckpt --input-device ReSpeaker \
        --channels 4 --seconds 120 --out live_out.wav --json live_rtf.json
Benoetigt: pip install sounddevice   (PortAudio; unter Linux ggf. libportaudio2)
"""
import argparse
import json
import os
import queue
import sys
import time

import numpy as np
import soundfile as sf
import torch

from measure_rtf_blockwise import HERE, TimedVAD, load_beats, stats   # importiert auch nbss_separator / BEATs
from nbss_separator import NBSSSeparator
import anonymizer
from stream_anonymizer import StreamAnonymizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--input-device", default=None, help="Index oder Namensteil, z. B. ReSpeaker")
    ap.add_argument("--channels", type=int, default=4, help="Kanaele, die das Geraet liefert")
    ap.add_argument("--use-channels", default=None, help="Kommagetrennte Geraetekanaele fuer das Modell (4 Stueck, "
                    "Reihenfolge wie im Training, Referenz = erster), Standard 0,1,2,3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--block", type=float, default=2.0)
    ap.add_argument("--ctx", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=1024, help="Samples je Mikrofon-Callback")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=1, help="erste Bloecke, die aus der Statistik fallen")
    ap.add_argument("--max-backlog", type=float, default=20.0, help="Abbruch, wenn so viele Sekunden Audio warten")
    ap.add_argument("--out", default="live_out.wav")
    ap.add_argument("--json")
    ap.add_argument("--beats-ckpt", default=os.path.join(HERE, "beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"))
    ap.add_argument("--ontology", default=os.path.join(HERE, "utils/audioset_ontology.json"))
    ap.add_argument("--speech-ths", type=float, default=0.3)
    ap.add_argument("--segmin", type=float, default=2.0)
    ap.add_argument("--segmax", type=float, default=2.0)
    ap.add_argument("--seghopmin", type=float, default=0.95)
    ap.add_argument("--seghopmax", type=float, default=0.95)
    ap.add_argument("--framemin", type=float, default=0.0)
    ap.add_argument("--framehopmin", type=float, default=0.95)
    ap.add_argument("--mixframe", type=int, default=0)
    ap.add_argument("--reverse", default="backward")
    ap.add_argument("--method", default="random_splicing_reverse")
    args = ap.parse_args()

    import sounddevice as sd
    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.ckpt:
        raise SystemExit("--ckpt fehlt")

    dev = torch.device(args.device)
    sep = NBSSSeparator(args.ckpt, device=args.device, use_bf16=not args.fp32, seg=None)
    sr = sep.sr
    beats, speech_idx = load_beats(args.beats_ckpt, args.ontology, dev)
    vad = TimedVAD(beats, dev)

    t_splice = []
    _orig = anonymizer.blurring_random_splicing

    def timed_splice(*a, **k):
        t0 = time.perf_counter()
        r = _orig(*a, **k)
        t_splice.append(time.perf_counter() - t0)
        return r
    anonymizer.blurring_random_splicing = timed_splice

    kw = dict(segmin=int(sr * args.segmin), segmax=int(sr * args.segmax), framemin=int(sr * args.framemin),
              seghopmin_perc=args.seghopmin, seghopmax_perc=args.seghopmax, framehopmin_perc=args.framehopmin,
              mixframe=bool(args.mixframe), reverse=args.reverse, device=dev, vad_model=vad, vad_sr=16000,
              vad_speech_index=speech_idx, vad_speech_ths=args.speech_ths, method=args.method)

    # Warm-up (CUDA-Init, Kernel-Auswahl) mit Rauschen, getrennte Instanz, damit der Live-Strom unberuehrt bleibt
    warm = StreamAnonymizer(sep, kw, blur_fn=anonymizer.blur_audio, block_s=args.block, ctx_s=args.ctx)
    rng = np.random.default_rng(0)
    for _ in range(3):
        warm.push(0.01 * rng.standard_normal((sep.N_CHN, warm.block)).astype(np.float32))
    vad.t.clear()
    t_splice.clear()
    del warm

    anon = StreamAnonymizer(sep, kw, blur_fn=anonymizer.blur_audio, block_s=args.block, ctx_s=args.ctx)
    use = [int(c) for c in args.use_channels.split(",")] if args.use_channels else list(range(sep.N_CHN))
    assert len(use) == sep.N_CHN and max(use) < args.channels, (use, args.channels)
    dev_in = int(args.input_device) if (args.input_device or "").isdigit() else args.input_device

    q = queue.Queue()
    events = []

    def callback(indata, frames, time_info, status):
        if status:
            events.append((time.monotonic(), str(status)))
        q.put((time.monotonic(), indata[:, use].T.copy()))

    rows, backlog_max = [], 0.0
    dur = anon.block_seconds
    print(f"Block={anon.block} Samples = {dur:.3f} s, Kontext={args.ctx} s, Chunk={args.chunk} Samples "
          f"({1000 * args.chunk / sr:.0f} ms), NBSS={'fp32' if args.fp32 else 'bf16'}, Kanaele {use}")
    print(f"Aufnahme {args.seconds:.0f} s ... (Strg+C beendet frueher)")
    out = sf.SoundFile(args.out, "w", samplerate=sr, channels=1)
    stream = sd.InputStream(device=dev_in, channels=args.channels, samplerate=sr, dtype="float32",
                            blocksize=args.chunk, callback=callback)
    t_stop = None
    try:
        with stream:
            t_stop = time.monotonic() + args.seconds
            while time.monotonic() < t_stop:
                try:
                    t_arr, chunk = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                backlog = q.qsize() * args.chunk / sr
                backlog_max = max(backlog_max, backlog)
                i_vad, i_spl = len(vad.t), len(t_splice)
                res = anon.push(chunk)
                for y in res:
                    out.write(y)
                    tot, tsep = anon.t_block[-1], anon.ctxsep.t_sep[-1]
                    tb, ts = sum(vad.t[i_vad:]), sum(t_splice[i_spl:])
                    rows.append(dict(total=tot, nbss=tsep, beats=tb, splice=ts, rest=tot - tsep - tb - ts,
                                     latency=time.monotonic() - t_arr, backlog=backlog))
                if backlog > args.max_backlog:
                    print(f"[!] Backlog {backlog:.1f} s > {args.max_backlog} s: Verarbeitung ist langsamer als Echtzeit")
                    break
    except KeyboardInterrupt:
        print("abgebrochen")
    finally:
        out.close()

    keep = rows[args.warmup:]
    print(f"\n{len(rows)} Bloecke ausgegeben ({len(keep)} in der Statistik), {anon.pending_seconds:.2f} s Rest im Puffer verworfen")
    if keep:
        rtf = np.array([r["total"] for r in keep]) / dur
        for k, name in [("total", "gesamt"), ("nbss", "NBSS (Ctx+Block)"), ("beats", "BEATs"),
                        ("splice", "Random Splicing"), ("rest", "Rest"), ("latency", "Latenz ab Blockende")]:
            print(f"  {name:<20s} {stats([r[k] for r in keep])}")
        print(f"\n  RTF  Mittel {rtf.mean():.3f}  p95 {np.percentile(rtf, 95):.3f}  MAX {rtf.max():.3f}   "
              f"(Bloecke mit RTF >= 1: {(rtf >= 1).sum()})")
    print(f"  Backlog max: {backlog_max:.2f} s   Eingabe-Meldungen von PortAudio: {len(events)}")
    for t, s in events[:5]:
        print(f"    {s}")
    ok = bool(keep) and rtf.max() < 1 and not events
    print("  -> " + ("Kriterium erfuellt (RTF < 1, keine Ueberlaeufe)" if ok else "Kriterium NICHT erfuellt, Details oben"))
    print(f"Ausgabe (nur anonymisiert): {args.out}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(block_s=dur, rows=rows, backlog_max=backlog_max, events=[s for _, s in events]), f)


if __name__ == "__main__":
    main()