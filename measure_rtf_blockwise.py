"""RTF-Messung Weg A (Ringpuffer, Block + Kontext) mit Zeiten je Stufe.

Spielt eine 4-Kanal-Datei als Strom in kleinen Chunks ein (ohne zu warten, nur Rechenzeit zaehlt) und misst
je Block: gesamt, NBSS (inkl. Kontext), BEATs, Random Splicing, Rest (Chunking, Resampling, Concat, ...).
Kriterium: Rechenzeit pro Block < Blockdauer  (RTF < 1); entscheidend ist der schlechteste Block.

Beispiel (im Repo-Root, neben nbss_separator.py, anonymizer.py, stream_anonymizer.py):
  python measure_rtf_blockwise.py --ckpt nbss/ckpts/epoch70_metric23.9507.ckpt \
      --wav /project/brunke/nbss_runs/test_in/test_000000_mixture.wav --device cuda --seconds 60 --out out.wav
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
from nbss_separator import NBSSSeparator            # setzt den ./nbss-Pfad und das GPU-Speicherlimit
sys.path.insert(0, os.path.join(HERE, "beats"))
from BEATs import BEATs, BEATsConfig                # noqa: E402
import anonymizer                                   # noqa: E402
from stream_anonymizer import StreamAnonymizer      # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


class TimedVAD:
    """Proxy: blur_audio ruft nur .extract_features() auf."""

    def __init__(self, model, dev):
        self.model, self.dev, self.t = model, dev, []

    def extract_features(self, *a, **k):
        sync(self.dev)
        t0 = time.perf_counter()
        r = self.model.extract_features(*a, **k)
        sync(self.dev)
        self.t.append(time.perf_counter() - t0)
        return r


def load_beats(path, ontology, dev):
    ckpt = torch.load(path, map_location=dev)
    m = BEATs(BEATsConfig(ckpt["cfg"]))
    m.load_state_dict(ckpt["model"])
    m.eval().to(dev)
    with open(ontology) as f:
        name_to_id = {e["name"]: e["id"] for e in json.load(f)}
    speech_id = name_to_id["Speech"]
    idx = next(k for k, v in ckpt["label_dict"].items() if v == speech_id)
    return m, idx


def stats(v):
    v = np.asarray(v) * 1000
    return f"Median {np.median(v):7.1f}  p95 {np.percentile(v, 95):7.1f}  max {v.max():7.1f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="NBSS in fp32 statt bf16-Autocast")
    ap.add_argument("--block", type=float, default=2.0, help="gewuenschte Blocklaenge in s (wird auf BEATs-Raster gerundet)")
    ap.add_argument("--ctx", type=float, default=2.0, help="Links-Kontext fuer NBSS in s")
    ap.add_argument("--input-chunk", type=int, default=1024, help="Samples je simuliertem Mikro-Callback")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--loop", action="store_true", help="kurze Datei wiederholen, bis --seconds erreicht sind")
    ap.add_argument("--warmup", type=int, default=3, help="Warm-up-Bloecke, die verworfen werden")
    ap.add_argument("--out", help="anonymisierte Ausgabe als wav")
    ap.add_argument("--json", help="Messwerte je Block als json")
    ap.add_argument("--beats-ckpt", default=os.path.join(HERE, "beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"))
    ap.add_argument("--ontology", default=os.path.join(HERE, "utils/audioset_ontology.json"))
    ap.add_argument("--speech-ths", type=float, default=0.3)
    # Anonymisierungs-Parameter (Defaults = anonymize_audio_folder.py)
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

    dev = torch.device(args.device)
    sep = NBSSSeparator(args.ckpt, device=args.device, use_bf16=not args.fp32, seg=None)
    sr = sep.sr
    beats, speech_idx = load_beats(args.beats_ckpt, args.ontology, dev)
    vad = TimedVAD(beats, dev)

    # Zeit fuer Random Splicing mitzaehlen (blur_audio sucht die Funktion im Modul)
    t_splice = []
    _orig_splice = anonymizer.blurring_random_splicing

    def timed_splice(*a, **k):
        t0 = time.perf_counter()
        r = _orig_splice(*a, **k)
        t_splice.append(time.perf_counter() - t0)
        return r
    anonymizer.blurring_random_splicing = timed_splice

    kw = dict(segmin=int(sr * args.segmin), segmax=int(sr * args.segmax),
              framemin=int(sr * args.framemin), seghopmin_perc=args.seghopmin, seghopmax_perc=args.seghopmax,
              framehopmin_perc=args.framehopmin, mixframe=bool(args.mixframe), reverse=args.reverse,
              device=dev, vad_model=vad, vad_sr=16000, vad_speech_index=speech_idx,
              vad_speech_ths=args.speech_ths, method=args.method)
    anon = StreamAnonymizer(sep, kw, blur_fn=anonymizer.blur_audio, block_s=args.block, ctx_s=args.ctx)
    assert args.input_chunk <= anon.block

    x, fs = sf.read(args.wav, dtype="float32", always_2d=True)
    assert fs == sr and x.shape[1] >= sep.N_CHN, (fs, x.shape)
    x = x.T[:sep.N_CHN]
    if args.loop and x.shape[1] < int(args.seconds * sr):
        x = np.tile(x, (1, int(np.ceil(args.seconds * sr / x.shape[1]))))
    x = x[:, :int(args.seconds * sr)]
    print(f"Geraet={dev}, Eingang {x.shape[1] / sr:.1f} s, Block={anon.block} Samples = {anon.block_seconds:.3f} s "
          f"(Raster), Kontext={args.ctx} s, NBSS-Praezision={'bf16' if sep.use_bf16 else 'fp32'}")

    rows, outs = [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in range(0, x.shape[1], args.input_chunk):
        i_vad, i_spl = len(vad.t), len(t_splice)
        res = anon.push(x[:, s:s + args.input_chunk])
        if not res:
            continue
        assert len(res) == 1
        outs.append(res[0])
        tot, tsep = anon.t_block[-1], anon.ctxsep.t_sep[-1]
        tb, ts = sum(vad.t[i_vad:]), sum(t_splice[i_spl:])
        rows.append(dict(total=tot, nbss=tsep, beats=tb, splice=ts, rest=tot - tsep - tb - ts))

    keep = rows[args.warmup:]
    if not keep:
        raise SystemExit("zu wenig Bloecke fuer die Messung (--seconds erhoehen oder --warmup senken)")
    dur = anon.block_seconds
    rtf = np.array([r["total"] for r in keep]) / dur
    print(f"\n{len(rows)} Bloecke, davon {len(keep)} gemessen (Warm-up {args.warmup} verworfen); Blockdauer {dur:.3f} s")
    for k, name in [("total", "gesamt"), ("nbss", "NBSS (Ctx+Block)"), ("beats", "BEATs"),
                    ("splice", "Random Splicing"), ("rest", "Rest")]:
        print(f"  {name:<18s} {stats([r[k] for r in keep])}")
    print(f"\n  RTF  Mittel {rtf.mean():.3f}  Median {np.median(rtf):.3f}  p95 {np.percentile(rtf, 95):.3f}  MAX {rtf.max():.3f}")
    print(f"  Bloecke mit RTF >= 1: {(rtf >= 1).sum()} von {len(rtf)}")
    print(f"  Latenz Eingang->Ausgabe (schlechtester Block): {dur + rtf.max() * dur:.2f} s")
    if dev.type == "cuda":
        print(f"  GPU-Spitzenspeicher: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"  im Puffer verbleibend (nie ausgegeben): {anon.pending_seconds:.2f} s")

    if args.out:
        sf.write(args.out, np.concatenate(outs), sr)
        print(f"Ausgabe: {args.out}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(block_s=dur, rows=rows, warmup=args.warmup), f)


if __name__ == "__main__":
    main()