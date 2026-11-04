"""
Tests fuer mc_anonymizer.py. Im Repo-Root ausfuehren:
    python test_mc_anonymizer.py
Laeuft ohne GPU/Checkpoints; fehlen torch/speechbrain, werden sie gestubbt.
"""
import random
import sys
import types

import numpy as np

try:
    import anonymizer  # noqa: F401
except ImportError:
    for name in ["torch", "torchaudio", "torchaudio.transforms", "torchaudio.pipelines",
                 "speechbrain", "speechbrain.inference", "speechbrain.inference.interfaces",
                 "torch.nn", "torch.nn.functional"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["torchaudio.transforms"].Fade = object
    sys.modules["torchaudio.pipelines"].HDEMUCS_HIGH_MUSDB_PLUS = None
    sys.modules["torchaudio.pipelines"].CONVTASNET_BASE_LIBRI2MIX = None
    sys.modules["speechbrain.inference.interfaces"].Pretrained = object
    sys.modules["torch"].Tensor = np.ndarray
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]

from anonymizer import blurring_random_splicing  # noqa: E402
from mc_anonymizer import blur_audio_mc, make_plan, apply_plan  # noqa: E402

SR = 16000
PARAMS = dict(segmin=2 * SR, segmax=2 * SR, framemin=0, seghopmin_perc=0.95,
              seghopmax_perc=0.95, framehopmin_perc=0.95)


def synth_voice(T=10 * SR, seed=0):
    """Rauschen mit sprachaehnlicher Huellkurve (Bursts mit Pausen)."""
    r = np.random.RandomState(seed)
    env = np.zeros(T)
    t = 0
    while t < T:
        on, off = r.randint(2000, 9000), r.randint(2000, 6000)
        env[t:t + on] = 0.2 + 0.8 * r.rand()
        t += on + off
    return (r.randn(T) * env).astype(np.float32)


def gcc_phat_lag(a, b, max_lag=20):
    n = len(a) + len(b)
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    R = A * np.conj(B)
    cc = np.fft.irfft(R / (np.abs(R) + 1e-12), n)
    cc = np.concatenate((cc[-max_lag:], cc[:max_lag + 1]))
    return int(np.argmax(cc) - max_lag)


def median_lag(x0, x1, win=4000):
    lags = []
    for s in range(0, len(x0) - win, win):
        if np.std(x0[s:s + win]) > 1e-3 and np.std(x1[s:s + win]) > 1e-3:
            lags.append(gcc_phat_lag(x0[s:s + win], x1[s:s + win]))
    return int(np.median(lags))


def run(voice4, other4, **kw):
    mix = voice4 + other4

    def sep(x):  # Oracle: Kanal 0 von x ist der gerollte Kanal
        # finde Rolle anhand Gleichheit mit mix
        for c in range(len(mix)):
            if np.array_equal(np.roll(mix, -c, axis=0), x):
                return np.roll(voice4, -c, axis=0)[0], np.roll(other4, -c, axis=0)[0]
        raise AssertionError("unbekannte Rollung")

    kw.setdefault("tail_mode", "last_window")
    return blur_audio_mc(mix, SR, sep_ref0=sep, vad_model=None, **PARAMS, **kw)


def test_mono_equivalence():
    v = synth_voice()
    voice4 = np.stack([v, 0.5 * v, 0.8 * v, 0.3 * v])
    other4 = np.zeros_like(voice4)
    out, _ = run(voice4, other4, mixframe=True, reverse="backward", seed=123, tail_mode="raw")
    out_tail, _ = run(voice4, other4, mixframe=True, reverse="backward", seed=123, tail_mode="last_window")
    random.seed(123)
    np.random.seed(123)
    mono = blurring_random_splicing(v.copy(), SR, PARAMS["segmin"], PARAMS["segmax"], 0.95, 0.95,
                                    p_reverse=1.0, framehopmin_perc=0.95, mixframe=True, framemin=0)
    mono = np.pad(mono, (0, max(0, len(v) - len(mono))))[:len(v)]
    n = 155000  # nur der von Fenstern abgedeckte Bereich (Rest hinter dem letzten Fenster separat)
    assert np.allclose(out[0][:n], mono[:n], atol=1e-4), np.abs(out[0][:n] - mono[:n]).max()
    print("OK  Kanal 0 == Mono-Pipeline (gleicher Seed, tail_mode=raw, ohne Rest)")
    leak = np.abs(out[0][157000:] - mono[157000:len(v)]).max()
    assert np.allclose(out_tail[0], mono[:len(v)], atol=1e-4)
    print(f"OK  tail_mode=last_window anonymisiert auch den Rest (raw wuerde Abweichung {leak:.2f} zeigen = Rohsprache)")


def test_shared_plan_gain():
    v = synth_voice(seed=1)
    g = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float32)[:, None]
    out, _ = run(g * v, np.zeros((4, len(v)), np.float32), mixframe=True, reverse="random", seed=7)
    for c in range(1, 4):
        assert np.allclose(out[c], g[c, 0] * out[0], atol=1e-5)
    print("OK  Pegelverhaeltnisse zwischen Kanaelen bleiben exakt erhalten")


def test_tdoa_under_reversal():
    v = synth_voice(seed=2)
    d = 3
    voice4 = np.stack([v, np.roll(v, d), np.roll(v, -d), v])
    other = np.zeros_like(voice4)
    in_lag = median_lag(voice4[0], voice4[1])
    res = {}
    for rev in ["forward", "backward"]:
        out, _ = run(voice4, other, mixframe=False, reverse=rev, seed=5)
        res[rev] = median_lag(out[0], out[1])
    print(f"    TDOA ch0-ch1: Eingang {in_lag}, forward {res['forward']}, backward {res['backward']}")
    assert res["forward"] == in_lag
    assert res["backward"] == -in_lag
    print("OK  Reversal spiegelt den TDOA der Sprache (Vorzeichenwechsel), forward erhaelt ihn")


if __name__ == "__main__":
    test_mono_equivalence()
    test_shared_plan_gain()
    test_tdoa_under_reversal()