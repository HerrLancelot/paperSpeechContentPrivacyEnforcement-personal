"""
Aequivalenztest: Kanal 0 der Mehrkanal-Ausgabe (StreamAnonymizerMC) == Ausgabe der bisherigen Mono-Kette
(StreamAnonymizer + blur_audio) bei gleichen Seeds. Laeuft auf dem Laptop ohne GPU und ohne Checkpoint:
der Separator ist ein Fake (Sprache = 0.8 * Referenzkanal, Hintergrund = 0.2 * Referenzkanal).

Im Repo-Root (neben anonymizer.py, stream_anonymizer.py, stream_anonymizer_mc.py, mc_anonymizer.py):
  python test_equivalence_mono.py                       # synthetisches Signal, Energie-VAD (schnell)
  python test_equivalence_mono.py --wav x_4ch_16k.wav --real-vad    # echte Aufnahme, echtes BEATs auf der CPU

--blur adapter: nur noetig, falls anonymizer.blur_audio kein [C, T] nimmt (unveraenderte Upstream-Version).
"""
import argparse
import random
import sys
import types

import numpy as np

SR = 16000


# --------------------------------------------------------------------------------------
# Falls torch & Co. fehlen (nur in Umgebungen ohne Installation): kleines numpy-Shim
# --------------------------------------------------------------------------------------
def _install_shim():
    class FT(np.ndarray):
        def clone(self): return self.copy().view(FT)
        def detach(self): return self
        def to(self, *a, **k): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
        def bool(self): return self.astype(bool).view(FT)
        def size(self, i=None): return self.shape if i is None else self.shape[i]

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch = types.ModuleType("torch")
    torch.tensor = lambda x, dtype=None: np.array(x, dtype=np.float32).view(FT)
    torch.float32 = np.float32
    torch.max, torch.abs = np.max, np.abs
    torch.zeros = lambda *s: np.zeros(s, dtype=np.float32).view(FT)
    torch.no_grad = _NoGrad
    torch.Tensor = np.ndarray
    nn, fn = types.ModuleType("torch.nn"), types.ModuleType("torch.nn.functional")
    fn.pad = lambda x, pad: np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad[0], pad[1])]).view(FT)
    nn.functional, torch.nn = fn, nn
    ta, tr, pl = (types.ModuleType(n) for n in ("torchaudio", "torchaudio.transforms", "torchaudio.pipelines"))

    class Resample:
        def __init__(self, orig_freq, new_freq):
            assert orig_freq == new_freq
        def __call__(self, x): return x
    tr.Resample, tr.Fade = Resample, object
    pl.HDEMUCS_HIGH_MUSDB_PLUS = pl.CONVTASNET_BASE_LIBRI2MIX = None
    ta.transforms, ta.pipelines = tr, pl
    sb, sbi, sbii = (types.ModuleType(n) for n in ("speechbrain", "speechbrain.inference",
                                                   "speechbrain.inference.interfaces"))
    sbii.Pretrained = object
    for name, m in [("torch", torch), ("torch.nn", nn), ("torch.nn.functional", fn), ("torchaudio", ta),
                    ("torchaudio.transforms", tr), ("torchaudio.pipelines", pl), ("speechbrain", sb),
                    ("speechbrain.inference", sbi), ("speechbrain.inference.interfaces", sbii)]:
        sys.modules[name] = m


try:
    import anonymizer  # noqa: F401
    SHIM = False
except ImportError:
    _install_shim()
    SHIM = True
    import anonymizer  # noqa: F401

import torch  # noqa: E402

from stream_anonymizer import StreamAnonymizer  # noqa: E402
from stream_anonymizer_mc import StreamAnonymizerMC  # noqa: E402


class FakeSep:
    sr, N_CHN, use_bf16 = SR, 4, False

    def separate(self, x):
        return 0.8 * x[0], 0.2 * x[0]


class EnergyVAD:
    """Deterministischer Ersatz fuer BEATs: 'Sprache', wenn die Fensterenergie hoch genug ist."""

    def extract_features(self, x, padding_mask=None):
        a = x.detach().cpu().numpy()
        logits = np.zeros((a.shape[0], 3), np.float32)
        logits[:, 1] = np.log10(np.mean(a ** 2, axis=1) + 1e-9) + 2.3
        return (torch.tensor(logits),)


def synth_input(n_blocks, block):
    r = np.random.RandomState(4)
    T = n_blocks * block
    env = np.zeros(T)
    t = 0
    while t < T:
        on, off = r.randint(4000, 14000), r.randint(3000, 9000)
        env[t:t + on] = 0.2 + 0.8 * r.rand()
        t += on + off
    v = (r.randn(T) * env).astype(np.float32)
    g = np.array([1.0, 0.5, 0.8, 0.3], np.float32)[:, None]
    return g * v


def main(a):
    if SHIM:
        print("(torch/speechbrain nicht installiert: numpy-Shim aktiv)")
    sep = FakeSep()
    if a.real_vad:
        from measure_rtf_blockwise import load_beats
        vad, vad_idx = load_beats(a.beats_ckpt, a.ontology, torch.device("cpu"))
    else:
        vad, vad_idx = EnergyVAD(), 1
    kw = dict(segmin=2 * SR, segmax=2 * SR, framemin=0, seghopmin_perc=0.95, seghopmax_perc=0.95,
              framehopmin_perc=0.95, mixframe=False, reverse="backward", device="cpu", vad_model=vad,
              vad_sr=SR, vad_speech_index=vad_idx, vad_speech_ths=0.3, method="random_splicing_reverse")

    if a.blur == "adapter":
        def blur_fn(blk, sr, source_sep, **k):
            class Ad:
                def separate(self, y1d, sr_):
                    return source_sep.separate(blk, sr_)
            return anonymizer.blur_audio(blk[0], sr, source_sep=Ad(), **k)
    else:
        blur_fn = anonymizer.blur_audio

    mono = StreamAnonymizer(sep, kw, blur_fn=blur_fn)
    mc = StreamAnonymizerMC(sep, kw, seed=a.seed)
    assert mono.block == mc.block, (mono.block, mc.block)
    blk = mono.block

    if a.wav:
        import soundfile as sf
        x, fs = sf.read(a.wav, dtype="float32", always_2d=True)
        assert fs == SR and x.shape[1] >= 4
        x = x.T[:4]
    else:
        x = synth_input(4, blk)
    n_blocks = x.shape[1] // blk
    print(f"Block {blk} Samples, {n_blocks} Bloecke, VAD: {'BEATs' if a.real_vad else 'Energie-Fake'}")

    worst = 0.0
    for i in range(n_blocks):
        b = x[:, i * blk:(i + 1) * blk]
        random.seed(a.seed + i)
        np.random.seed(a.seed + i)
        y_mono = mono.push(b)
        y_mc = mc.push(b)
        assert len(y_mono) == 1 and len(y_mc) == 1
        d = float(np.abs(y_mono[0] - y_mc[0][0]).max())
        worst = max(worst, d)
        changed = float(np.mean(np.abs(y_mc[0][0] - b[0]) > 1e-6))
        print(f"  Block {i + 1}: max |mono - mc[0]| = {d:.2e}, Anteil veraendert gegenueber Rohsignal {changed:.2f}")
        if not a.wav:
            for c in range(1, 4):
                g = x[c, 0] / x[0, 0] if x[0, 0] else None
                ratio = np.abs(y_mc[0][c]).max() / max(np.abs(y_mc[0][0]).max(), 1e-12)
                assert abs(ratio - np.abs(x[c]).max() / np.abs(x[0]).max()) < 0.05 or g is None, (c, ratio)
    ok = worst < 1e-4
    print("OK  Kanal 0 der Mehrkanal-Kette == Mono-Kette" if ok else f"FEHLER: max. Abweichung {worst:.2e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav", help="4-Kanal-WAV, 16 kHz (sonst synthetisches Signal)")
    p.add_argument("--real-vad", action="store_true", help="echtes BEATs auf der CPU statt Energie-VAD")
    p.add_argument("--beats-ckpt", default="beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    p.add_argument("--ontology", default="utils/audioset_ontology.json")
    p.add_argument("--blur", choices=["native", "adapter"], default="native")
    p.add_argument("--seed", type=int, default=5)
    main(p.parse_args())