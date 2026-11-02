"""Gleichheitstest: Parallel (Training/separate()) vs. Frame-fuer-Frame mit persistentem Zustand.

Test 1 (arch)  : gleiche Eingabe X [B,F,T,2C] -> arch(X) vs. ArchStreamer.step() pro Frame
Test 2 (e2e)   : Wellenform -> STFT -> Norm -> arch -> iNorm -> iSTFT, parallel vs. blockweise (1 Hop)
                 (+ Zeit pro Block, nur zur ersten Orientierung)

Aufruf (Cluster, im NBSS-Repo, training venv):
  python test_streaming_equality.py --ckpt /pfad/epoch36_metric23.0607.ckpt --wav clip.wav --device cuda
Ohne --ckpt: zufaellig initialisiertes Modell (prueft nur die Logik, nicht dein Modell).
Fallback, falls Lightning das Modell nicht selbst rekonstruiert:
  --arch-kwargs '{"dim_input":8,"dim_output":4,...}'
"""
import argparse
import contextlib
import json
import os
import sys
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "nbss"))
sys.path.insert(0, HERE)

from models.arch.OnlineSpatialNet import OnlineSpatialNet
from models.io.norm import Norm
from models.io.stft import STFT
from streaming_ospnet import ArchStreamer, StreamISTFT, StreamSTFT

RANDOM_KWARGS = dict(dim_input=8, dim_output=4, num_layers=8, dim_hidden=96, dim_ffn=192, num_heads=4,
                     kernel_size=(5, 3), conv_groups=(8, 8), norms=["LN", "LN", "GN", "LN", "LN", "LN"],
                     dim_squeeze=8, num_freqs=257, full_share=0, attention="ret(2)",
                     decay=[4, 5, 9, 10], rope=False)


def load_model(args):
    if args.ckpt is None:
        print("[!] kein Checkpoint: zufaellige Gewichte (nur Logik-Test)")
        torch.manual_seed(0)
        arch = OnlineSpatialNet(**RANDOM_KWARGS)
        return arch, STFT(512, 256, 512), Norm("utterance", online=True), [0, 1, 2, 3], 0
    if args.separator_dir:
        # exakt dasselbe Laden wie in der Pipeline (strict=True, Config 1:1 aus nbss_separator.py)
        sys.path.insert(0, os.path.abspath(args.separator_dir))
        if not torch.cuda.is_available():  # nbss_separator.py ruft beim Import set_per_process_memory_fraction auf
            torch.cuda.set_per_process_memory_fraction = lambda *a, **k: None
        from nbss_separator import NBSSSeparator
        sep = NBSSSeparator(args.ckpt, "cpu", use_bf16=False, seg=None)
        print("[ok] Modell ueber NBSSSeparator geladen (strict=True)")
        return sep.arch, sep.stft, sep.norm, list(range(sep.N_CHN)), sep.REF_CHANNEL
    try:
        from SharedTrainer import SharedTrainer
        m = SharedTrainer.load_from_checkpoint(args.ckpt, map_location="cpu")
        print(f"[ok] Checkpoint ueber SharedTrainer geladen; stft={m.stft}, norm={m.norm}, channels={m.channels}, ref={m.ref_channel}")
        if m.loss.mask is not None:
            print("[!] loss.mask ist nicht None -> Test setzt direkte Spektrum-Schaetzung voraus")
        return m.arch, m.stft, m.norm, list(m.channels), m.channels.index(m.ref_channel)
    except Exception as e:
        if args.arch_kwargs is None:
            raise SystemExit(f"load_from_checkpoint fehlgeschlagen ({type(e).__name__}: {e}). "
                             f"Bitte --arch-kwargs '<json>' angeben.")
        print(f"[!] load_from_checkpoint fehlgeschlagen ({type(e).__name__}); nutze --arch-kwargs")
        kw = json.loads(args.arch_kwargs)
        arch = OnlineSpatialNet(**kw)
        sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
        arch.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("arch.")}, strict=True)
        print("[!] Annahmen: STFT 512/256, Norm utterance online, Kanaele 0-3, ref 0")
        return arch, STFT(512, 256, 512), Norm("utterance", online=True), [0, 1, 2, 3], 0


def load_audio(args, n_chan, sr=16000):
    if args.wav is None:
        print("[!] keine --wav: synthetisches 4-Kanal-Signal (fuer echte Zahlen bitte echten Clip nehmen)")
        g = torch.Generator().manual_seed(1)
        n = int(args.seconds * sr)
        base = torch.randn(n + 8, generator=g)
        base = torch.nn.functional.conv1d(base[None, None], torch.hann_window(32)[None, None] / 8, padding=16)[0, 0]
        base = base / base.abs().max() * 0.3
        return torch.stack([base[d:d + n] + 0.01 * torch.randn(n, generator=g) for d in (0, 1, 2, 3)][:n_chan])
    import soundfile as sf
    x, fs = sf.read(args.wav, always_2d=True, dtype="float32")
    assert fs == sr, f"Samplerate {fs} != {sr}"
    x = torch.from_numpy(x.T)[:n_chan]
    return x[:, :int(args.seconds * sr)]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def report(name, ref, est):
    d = (ref - est).float()
    r = ref.float()
    agree = 10 * torch.log10((r ** 2).sum() / ((d ** 2).sum() + 1e-30)).item()
    print(f"  {name:<34s} max|d|={d.abs().max().item():.2e}  rel.max={d.abs().max().item() / (r.abs().max().item() + 1e-30):.2e}  "
          f"Uebereinstimmung={agree:6.1f} dB")
    return agree


def drift(ref, est, dim, bins=5):
    T = ref.shape[dim]
    edges = [int(T * i / bins) for i in range(bins + 1)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        r, e = ref.narrow(dim, a, b - a).float(), est.narrow(dim, a, b - a).float()
        out.append(10 * torch.log10((r ** 2).sum() / (((r - e) ** 2).sum() + 1e-30)).item())
    print("  Fehler ueber die Zeit (dB je Fuenftel):", "  ".join(f"{v:6.1f}" for v in out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--separator-dir", help="Ordner mit nbss_separator.py (und nbss/); laedt das Modell wie die Pipeline")
    ap.add_argument("--bf16", action="store_true", help="arch unter bf16-Autocast (wie Training); Standard: fp32")
    ap.add_argument("--arch-kwargs")
    ap.add_argument("--wav")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--per-layer", action="store_true", help="Hidden-State-Abweichung je Layer")
    ap.add_argument("--repo-inference", action="store_true", help="zusaetzlich arch(x, inference=True) des Repos vergleichen (langsam)")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    dev = torch.device(args.device)
    arch, stft, norm, chans, ref_idx = load_model(args)

    def ac():
        if args.bf16 and dev.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    arch, stft, norm = arch.to(dev).eval(), stft.to(dev).eval(), norm.to(dev).eval()
    wav = load_audio(args, len(chans)).to(dev)
    x = wav[None]  # [1,C,T]
    hop = stft.n_hop
    print(f"Geraet={dev}, Signal={tuple(x.shape)} ({x.shape[-1] / 16000:.1f} s), n_fft={stft.n_fft}, hop={hop}")

    with torch.no_grad():
        # ---------- parallele Referenz ----------
        X, L = stft.stft(x)                                   # [1,C,F,T]
        Xn, (Xr, XrMM) = norm.norm(X.clone(), ref_channel=ref_idx)
        B, C, Fq, T = Xn.shape
        Xin = torch.view_as_real(Xn.permute(0, 2, 3, 1)).reshape(B, Fq, T, -1)

        store_par = []
        hooks = []
        if args.per_layer:
            for l in arch.layers:
                hooks.append(l.register_forward_hook(lambda m, i, o: store_par.append(o[0].detach().clone())))
        with ac():
            y_par = arch(Xin)                                 # [1,F,T,out], chunkwise-Pfad wie im Training
        for h in hooks:
            h.remove()

        # ---------- Test 1: Arch-Ebene ----------
        print("\nTest 1: arch parallel vs. Frame-fuer-Frame mit Zustand")
        st = ArchStreamer(arch)
        store_str = [[] for _ in arch.layers]
        hooks = []
        if args.per_layer:
            for k, l in enumerate(arch.layers):
                hooks.append(l.register_forward_hook(lambda m, i, o, k=k: store_str[k].append(o[0].detach().clone())))
        with ac():
            ys = [st.step(Xin[:, :, t:t + 1, :]) for t in range(T)]
        for h in hooks:
            h.remove()
        y_str = torch.cat(ys, dim=2)
        a1 = report("arch-Ausgabe", y_par, y_str)
        drift(y_par, y_str, dim=2)
        if args.per_layer:
            for k in range(len(arch.layers)):
                report(f"Layer {k} (Hidden)", store_par[k], torch.cat(store_str[k], dim=2))
        if args.repo_inference:
            t0 = time.time()
            with ac():
                y_rep = arch(Xin, inference=True)
            print(f"  (Repo-inference=True: {time.time() - t0:.1f} s)")
            report("parallel vs. Repo-inference", y_par, y_rep)
            report("Repo-inference vs. Streamer", y_rep, y_str)

        # ---------- Test 2: Ende-zu-Ende Wellenform ----------
        print("\nTest 2: Wellenform parallel vs. blockweise (1 Hop = %d Samples)" % hop)
        out = torch.view_as_complex(y_par.float().reshape(B, Fq, T, -1, 2)).permute(0, 3, 1, 2)  # [1,S,F,T]
        wav_par = stft.istft(norm.inorm(out, (Xr, XrMM)), L)                                    # [1,S,N]

        st.reset()
        sstft, sistft = StreamSTFT(stft), StreamISTFT(stft)
        outs, t_block, t_arch = [], [], []
        n_blocks = x.shape[-1] // hop
        for i in range(n_blocks):
            sync(dev)
            t0 = time.perf_counter()
            frames = sstft.push(x[0, :, i * hop:(i + 1) * hop])
            ta = 0.0
            if frames is not None:
                for k in range(frames.shape[-1]):
                    Xf = frames[..., k][None, :, :, None]                      # [1,C,F,1]
                    Xf, (Xrf, XrMMf) = norm.norm(Xf, ref_channel=ref_idx)
                    xin = torch.view_as_real(Xf.permute(0, 2, 3, 1)).reshape(1, Fq, 1, -1)
                    sync(dev)
                    t1 = time.perf_counter()
                    with ac():
                        yf = st.step(xin)
                    sync(dev)
                    ta += time.perf_counter() - t1
                    of = torch.view_as_complex(yf.float().reshape(1, Fq, 1, -1, 2)).permute(0, 3, 1, 2)
                    Y = norm.inorm(of, (Xrf, XrMMf))[0, :, :, 0]               # [S,F]
                    o = sistft.push(Y)
                    if o is not None:
                        outs.append(o)
            sync(dev)
            t_block.append(time.perf_counter() - t0)
            t_arch.append(ta)
        wav_str = torch.cat(outs, dim=-1)                                      # [S,N']
        N = wav_str.shape[-1]
        print(f"  Streaming lieferte {N} von {x.shape[-1]} Samples (Rest: Ende-Padding, hier nicht geflusht)")
        S = wav_str.shape[0]
        for s in range(S):
            report(f"Wellenform, Quelle {s}" + (" (Sprache)" if s == 0 else " (Hintergrund)" if s == 1 else ""), wav_par[0, s, :N], wav_str[s])
        drift(wav_par[0, :, :N], wav_str, dim=-1)

    # ---------- Zeiten (nur Orientierung) ----------
    tb = torch.tensor(t_block[args.warmup:]) * 1000
    ta = torch.tensor([v for v in t_arch[args.warmup:]]) * 1000
    budget = hop / 16000 * 1000
    print(f"\nZeit pro Block (Budget {budget:.1f} ms), Warm-up {args.warmup} Blocks verworfen:")
    print(f"  gesamt : Median {tb.median():.2f}  p95 {tb.quantile(0.95):.2f}  max {tb.max():.2f} ms  -> RTF(max)={tb.max() / budget:.2f}")
    print(f"  Modell : Median {ta.median():.2f}  p95 {ta.quantile(0.95):.2f}  max {ta.max():.2f} ms")
    print("\nRichtwert: >60 dB = praktisch identisch, 40-60 dB = pruefen (--per-layer), <40 dB = Fehler im Zustandspfad.")


if __name__ == "__main__":
    main()