"""Tests fuer stream_anonymizer_mc.py (ohne GPU/Checkpoints). Im Repo-Root: python test_stream_mc.py"""
import sys
import types

import numpy as np

try:
    import anonymizer  # noqa: F401
except ImportError:
    for name in ["torch", "torchaudio", "torchaudio.transforms", "torchaudio.pipelines", "speechbrain",
                 "speechbrain.inference", "speechbrain.inference.interfaces", "torch.nn", "torch.nn.functional"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["torchaudio.transforms"].Fade = object
    sys.modules["torchaudio.pipelines"].HDEMUCS_HIGH_MUSDB_PLUS = None
    sys.modules["torchaudio.pipelines"].CONVTASNET_BASE_LIBRI2MIX = None
    sys.modules["speechbrain.inference.interfaces"].Pretrained = object
    sys.modules["torch"].Tensor = np.ndarray
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]

from stream_anonymizer_mc import StreamAnonymizerMC  # noqa: E402
from test_mc_anonymizer import synth_voice  # noqa: E402

SR = 16000


class FakeSep:
    """Sprache = 0.8 * Referenzkanal, Hintergrund = 0.2 * Referenzkanal (merkt sich die Eingabelaengen)."""
    sr, N_CHN, use_bf16 = SR, 4, False

    def __init__(self):
        self.lengths = []

    def separate(self, x):
        self.lengths.append(x.shape[1])
        return 0.8 * x[0], 0.2 * x[0]

    def separate_batch(self, xs):
        self.lengths.append(xs.shape[2])
        return 0.8 * xs[:, 0], 0.2 * xs[:, 0]


KW = dict(segmin=2 * SR, segmax=2 * SR, framemin=0, seghopmin_perc=0.95, seghopmax_perc=0.95,
          framehopmin_perc=0.95, mixframe=False, reverse="backward", device="cpu", vad_model=None,
          vad_sr=SR, vad_speech_index=None, vad_speech_ths=0.3, method="random_splicing_reverse")


def make_input(n_blocks=3):
    v = synth_voice(T=36160 * n_blocks + 5000, seed=4)
    g = np.array([1.0, 0.5, 0.8, 0.3], np.float32)[:, None]
    return g * v, g


def run(x, chunk, **kw):
    sep = FakeSep()
    a = StreamAnonymizerMC(sep, KW, seed=11, **kw)
    outs = []
    for s in range(0, x.shape[1], chunk):
        outs += a.push(x[:, s:s + chunk])
    return a, sep, outs


def test_block_grid_and_context():
    x, _ = make_input()
    a, sep, outs = run(x, 1024)
    assert a.block == 36160 and abs(a.block_seconds - 2.26) < 1e-9
    assert len(outs) == 3 and all(o.shape == (4, 36160) for o in outs)
    assert abs(a.pending_seconds - 5000 / SR) < 1e-9
    # 4 gerollte Laeufe je Block; erster Block ohne Kontext, danach ctx(2 s) + Block
    assert sep.lengths[:4] == [36160] * 4 and sep.lengths[4:8] == [36160 + 32000] * 4
    print("OK  Blockraster 36160, Kontext 2 s ab Block 2, Rest bleibt im Puffer")


def test_chunk_invariance_and_batch():
    x, _ = make_input()
    _, _, o1 = run(x, 1024)
    _, _, o2 = run(x, 7777)
    _, _, o3 = run(x, 1024, batch_roll=True)
    for a, b, c in zip(o1, o2, o3):
        assert np.array_equal(a, b) and np.allclose(a, c, atol=1e-6)
    print("OK  Chunkgroesse egal, Batch == sequentiell")


def test_channels_and_no_raw_passthrough():
    x, g = make_input()
    _, _, outs = run(x, 1024)
    for i, o in enumerate(outs):
        for c in range(1, 4):
            assert np.allclose(o[c], g[c, 0] * o[0], atol=1e-5)
        blk = x[:, i * 36160:(i + 1) * 36160]
        assert np.abs(o - blk).max() > 0.5 * np.abs(blk).max()   # nicht das Rohsignal
    print("OK  Pegelverhaeltnisse bleiben erhalten, Ausgang ist nicht das Rohsignal")


if __name__ == "__main__":
    test_block_grid_and_context()
    test_chunk_invariance_and_batch()
    test_channels_and_no_raw_passthrough()