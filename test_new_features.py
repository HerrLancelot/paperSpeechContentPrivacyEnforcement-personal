"""Tests fuer die neuen Funktionen: --dtype, --skip-nonspeech, --raw-out, eval_context_dtype.py, Separator-API.

Laeuft auf dem Laptop (CPU), ohne Modell und ohne Mikrofon (Zufallsgewichte, Fake-VAD, Fake-Mikrofon):
  python test_new_features.py                          # alle Tests
  python test_new_features.py --only skip,raw          # Auswahl: api,eval,vad,skip,mc,raw
  python test_new_features.py --ckpt <ckpt>            # api-Test zusaetzlich mit echtem Checkpoint (fp16/bf16 nur auf GPU)
Erwartet am Ende: 'ALLE TESTS OK'. Bei einem Fehler die Ausgabe schicken.
"""
import argparse
import builtins
import os
import random
import sys
import tempfile
import threading
import time
import types

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SR, CPU = 16000, torch.device("cpu")
RESULTS = []


def ok(msg):
    RESULTS.append(msg)
    print("OK  " + msg, flush=True)


def rel_db(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-30))


# ------------------------------------------------------------------ Fakes
class FakeVAD:
    """BEATs-Ersatz: Logit = mittlere Betragsamplitude des (auf Max 1 normierten) Fensters; speech_index = 1."""
    def __init__(self):
        self.inputs = []

    def extract_features(self, x, padding_mask=None):
        self.inputs.append(x.detach().clone())
        return (x.abs().mean(dim=-1, keepdim=True).repeat(1, 3),)


class FakeSep:
    """NBSS-Ersatz: Sprache = Kanal 0, Hintergrund = 0. Merkt sich jede Eingabe (Kontext + Block)."""
    sr, N_CHN, use_bf16, precision, n_fp16_fallback = SR, 4, False, "fp32", 0

    def __init__(self):
        self.inputs = []

    def separate(self, x):
        self.inputs.append(np.array(x, copy=True))
        return x[0].copy(), np.zeros_like(x[0])


def make_kw(vad):
    return dict(segmin=2 * SR, segmax=2 * SR, framemin=0, seghopmin_perc=0.95, seghopmax_perc=0.95,
                framehopmin_perc=0.95, mixframe=False, reverse="backward", device=CPU, vad_model=vad, vad_sr=16000,
                vad_speech_index=1, vad_speech_ths=0.3, method="random_splicing_reverse")


def make_blocks(block, n_blocks=6, seed=0, burst_only_ch2_in=None):
    """4-Kanal-Blocke: leises Rauschen; Block 1 hat Sprach-Ersatz (Rechteck) in der Mitte, Block 4 am Ende, Block 5 ein Grenzfall."""
    rng = np.random.default_rng(seed)
    t = np.arange(block) / SR
    sq = (0.8 * np.sign(np.sin(2 * np.pi * 180 * t))).astype(np.float32)
    blocks = []
    for b in range(n_blocks):
        x = (0.01 * rng.standard_normal((4, block))).astype(np.float32)
        lo, hi = {1: (8000, 20000), 4: (block - 9600, block)}.get(b, (0, 0))
        if hi > lo:
            for c in range(4):
                x[c, lo:hi] += sq[lo:hi] * (1.0 - 0.1 * c)
        if burst_only_ch2_in == b:
            x[2, 10000:22000] += sq[10000:22000]
        if b == 5:   # Grenzfall: mittlerer Pegel (0,5) + kurzer Klick (1,0). Mit der Auffuell-Eigenheit "keine Sprache",
            #          ohne sie "Sprache": deckt auf, wenn Vorab-VAD und Mehrkanal-Kette unterschiedlich entscheiden
            x[:, 8000:24000] += 0.5 * np.sign(np.sin(2 * np.pi * 150 * t[8000:24000])).astype(np.float32)
            x[:, 30000:30050] += 1.0
        blocks.append(x)
    return blocks


def feed(anon, blocks, reseed):
    outs = []
    for k, blk in enumerate(blocks):
        if reseed:
            random.seed(1000 + k)
            np.random.seed(1000 + k)
        outs += anon.push(blk)
    return outs


def block_inputs(sep, blocks, block):
    """Ordnet die gemerkten Separator-Eingaben den Bloecken zu (Eingabe endet immer mit dem Block)."""
    found = {}
    for x in sep.inputs:
        for k, blk in enumerate(blocks):
            if x.shape[1] >= block and np.array_equal(x[:, -block:], blk):
                found[k] = x
    return found


# ------------------------------------------------------------------ Tests
def test_api(args):
    import nbss_separator as ns
    patched = None
    if args.ckpt is None:                    # Zufallsgewichte statt Checkpoint
        orig_load, orig_lsd = torch.load, ns.OnlineSpatialNet.load_state_dict
        torch.load = lambda *a, **k: {"state_dict": {}}
        ns.OnlineSpatialNet.load_state_dict = lambda self, sd, strict=True: None
        patched = (orig_load, orig_lsd)
    ckpt = args.ckpt or "dummy.ckpt"
    try:
        torch.manual_seed(0)
        sep = ns.NBSSSeparator(ckpt, "cpu", use_bf16=True, seg=None)
        assert sep.precision == "fp32" and sep.use_bf16 is False, sep.precision
        sep16 = ns.NBSSSeparator(ckpt, "cpu", precision="fp16", seg=None)
        assert sep16.precision == "fp32"
        ok("NBSSSeparator: auf CPU immer fp32 (use_bf16 und precision='fp16' werden still auf fp32 gesetzt)")
        try:
            ns.NBSSSeparator(ckpt, "cpu", precision="fp64")
            raise AssertionError("ValueError erwartet")
        except ValueError:
            ok("NBSSSeparator: ungueltige precision -> ValueError")
        rng = np.random.default_rng(0)
        x = (0.1 * rng.standard_normal((4, SR))).astype(np.float32)
        s, b = sep.separate(x)
        assert s.shape == b.shape == (SR,) and np.isfinite(s).all() and np.isfinite(b).all()
        rolls = np.stack([np.roll(x, -c, axis=0) for c in range(4)])
        sb, _ = sep.separate_batch(rolls)
        seq = np.stack([sep.separate(r)[0] for r in rolls])
        worst = min(rel_db(seq[i], sb[i]) for i in range(4))
        assert worst > 50, worst
        ok(f"separate / separate_batch: Form und Werte in Ordnung, Batch == einzeln ({worst:.0f} dB)")
        sep.use_bf16 = True                  # Setter muss existieren (Selbsttest nutzt ihn)
        sep.use_bf16 = False
        ok("use_bf16-Property (Kompatibilitaet)")
        if torch.cuda.is_available():
            gs = ns.NBSSSeparator(ckpt, "cuda", precision="fp32", seg=None)
            ref, _ = gs.separate(x)
            for prec in ("bf16", "fp16"):
                gs.precision = prec
                y, _ = gs.separate(x)
                assert np.isfinite(y).all(), f"{prec}: nicht endlich"
                msg = f"GPU {prec}: endlich, {rel_db(ref, y):.1f} dB zu fp32, fp16-Fallbacks {gs.n_fp16_fallback}"
                if args.ckpt is None:
                    msg += " (Zufallsgewichte, Zahl nicht aussagekraeftig)"
                ok(msg)
        else:
            print("[i] keine GPU: fp16/bf16-Pfad wird hier nicht getestet (Cluster: python nbss_separator.py ...)")
    finally:
        if patched:
            torch.load, ns.OnlineSpatialNet.load_state_dict = patched


def test_eval(args):
    from eval_context_dtype import run_blockwise, rel_db as erel, by_position, si_sdr

    class Fir:                                 # kausaler FIR: Ausgabe haengt nur von 99 vorherigen Samples ab
        sr, N_CHN, n_fp16_fallback = SR, 4, 0
        h = np.hanning(100).astype(np.float64)

        def separate(self, x):
            y = np.convolve(x[0], self.h)[:x.shape[1]].astype(np.float32)
            return y, np.zeros_like(y)

    from stream_anonymizer import snap_block
    block = snap_block(2.0, SR)
    x = np.random.default_rng(1).standard_normal((4, 6 * block)).astype(np.float32)
    ref, _ = Fir().separate(x)
    sp, _, times = run_blockwise(Fir(), x, block, ctx_s=0.5, sr=SR, warm=1)
    assert len(times) == 6 and len(sp) == 6 * block
    d = erel(ref[block:], sp[block:])
    assert d > 100, d
    ok(f"eval_context_dtype.run_blockwise: Kontext >= Filterlaenge -> identisch zur Ganzdatei ({d:.0f} dB)")
    sp0, _, _ = run_blockwise(Fir(), x, block, ctx_s=0.0, sr=SR, warm=0)
    d0 = erel(ref[block:], sp0[block:])
    assert d0 < 60, d0
    pos = by_position(ref, sp0, 6, block)
    assert pos[0] < pos[-1], pos
    ok(f"eval_context_dtype: ohne Kontext Abweichung ({d0:.0f} dB), Fehler konzentriert sich am Blockanfang "
       f"({pos[0]:.0f} dB im ersten, {pos[-1]:.0f} dB im letzten Fuenftel)")
    t = np.sin(np.arange(SR) * 0.05).astype(np.float64)
    assert si_sdr(t, t * 3) > 100
    ok("si_sdr: skalierungsinvariant")


def test_vad_helper(args):
    import anonymizer
    from stream_anonymizer import ContextSeparator, vad_flags_like_blur_audio, snap_block
    block = snap_block(2.0, SR)
    blk = make_blocks(block)[1]
    v1, v2 = FakeVAD(), FakeVAD()
    random.seed(0)
    np.random.seed(0)
    anonymizer.blur_audio(blk, SR, source_sep=ContextSeparator(FakeSep(), 0.0), **make_kw(v1))
    flags = vad_flags_like_blur_audio(blk[0], SR, v2, 16000, 1, 0.3, CPU, pad_quirk=True)
    assert torch.equal(v1.inputs[0], v2.inputs[0]), "BEATs-Eingabe der Vorab-VAD != blur_audio"
    n_win = v1.inputs[0].shape[0]
    assert v1.inputs[0].shape[-1] == 8000 + (8000 - n_win), v1.inputs[0].shape
    ok(f"Vorab-VAD: BEATs-Eingabe bitgleich zu blur_audio, Form {tuple(v1.inputs[0].shape)} "
       f"(= 8000 + {8000 - n_win} angehaengte Nullen, Auffuell-Eigenheit des Originals), Flags {flags.astype(int).tolist()}")
    return None


def _run_mono(skip, blocks, block):
    import anonymizer
    from stream_anonymizer import StreamAnonymizer
    sep = FakeSep()
    anon = StreamAnonymizer(sep, make_kw(FakeVAD()), blur_fn=anonymizer.blur_audio, block_s=2.0, ctx_s=2.0,
                            skip_nonspeech=skip)
    return anon, sep, feed(anon, blocks, reseed=True)


def test_skip_mono(args):
    from stream_anonymizer import snap_block
    block = snap_block(2.0, SR)
    blocks = make_blocks(block)
    a_off, s_off, o_off = _run_mono(False, blocks, block)
    a_on, s_on, o_on = _run_mono(True, blocks, block)
    assert len(o_off) == len(o_on) == len(blocks)
    worst = max(float(np.abs(a - b).max()) for a, b in zip(o_off, o_on))
    assert worst < 1e-5, f"Ausgabe mit --skip-nonspeech weicht ab: max|d|={worst:.2e}"
    assert 0 < a_on.n_skipped < len(blocks), a_on.n_skipped
    assert len(s_on.inputs) == len(blocks) - a_on.n_skipped and len(s_off.inputs) == len(blocks)
    ok(f"--skip-nonspeech (mono): Ausgabe identisch (max|d|={worst:.1e}), {a_on.n_skipped} von {len(blocks)} Bloecken "
       f"ohne NBSS ({len(s_on.inputs)} statt {len(s_off.inputs)} Aufrufe)")
    in_on, in_off = block_inputs(s_on, blocks, block), block_inputs(s_off, blocks, block)
    for k, x in in_on.items():
        assert np.array_equal(x, in_off[k]), f"Kontext in Block {k} stimmt nicht"
    ok(f"--skip-nonspeech (mono): Links-Kontext der Sprachbloecke {sorted(in_on)} identisch zum Lauf ohne Ueberspringen")
    assert len(a_on.t_block) == len(a_on.ctxsep.t_sep) == len(blocks)
    ok("--skip-nonspeech (mono): Zeitlisten haben je Block einen Eintrag")


def test_skip_mc(args):
    from stream_anonymizer import snap_block
    from stream_anonymizer_mc import StreamAnonymizerMC
    block = snap_block(2.0, SR)
    blocks = make_blocks(block, burst_only_ch2_in=3)
    for mode in ("ref", "any"):
        res = {}
        for skip in (False, True):
            sep = FakeSep()
            anon = StreamAnonymizerMC(sep, make_kw(FakeVAD()), block_s=2.0, ctx_s=2.0, vad_mode=mode, seed=7,
                                      skip_nonspeech=skip)
            res[skip] = (anon, sep, feed(anon, blocks, reseed=False))
        (a_off, s_off, o_off), (a_on, s_on, o_on) = res[False], res[True]
        assert a_on.block == block
        worst = max(float(np.abs(a - b).max()) for a, b in zip(o_off, o_on))
        assert worst < 1e-5, f"vad_mode={mode}: max|d|={worst:.2e}"
        assert 0 < a_on.n_skipped < len(blocks)
        assert len(s_on.inputs) < len(s_off.inputs)
        ok(f"--skip-nonspeech (--mc-out, vad-mode {mode}): Ausgabe identisch (max|d|={worst:.1e}), "
           f"{a_on.n_skipped} von {len(blocks)} Bloecken ohne NBSS ({len(s_on.inputs)} statt {len(s_off.inputs)} Laeufe)")
        if mode == "ref":
            n_ref = a_on.n_skipped
        else:
            assert a_on.n_skipped < n_ref, "Burst nur auf Kanal 2 muss im Modus 'any' erkannt werden"
            ok("vad-mode any erkennt Sprache, die nur auf Kanal 2 liegt (Block wird nicht uebersprungen)")


class FakeInputStream:
    """sounddevice.InputStream-Ersatz: spielt eine Matrix [n, channels] in Chunks (schneller als Echtzeit) ein."""
    audio, speed = None, 3.0

    def __init__(self, device=None, channels=4, samplerate=16000, dtype="float32", blocksize=1024, callback=None):
        self.cb, self.n, self.ch = callback, blocksize, channels
        self.t = None

    def __enter__(self):
        def run():
            for s in range(0, len(self.audio) - self.n + 1, self.n):
                self.cb(self.audio[s:s + self.n].copy(), self.n, None, None)
                time.sleep(self.n / SR / self.speed)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        return False


def test_raw(args):
    ont = os.path.join(HERE, "utils/audioset_ontology.json")
    beats = os.path.join(HERE, "beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    if not (os.path.exists(ont) and os.path.exists(beats)):
        print("[i] raw-Test uebersprungen: BEATs-Gewichte oder Ontologie fehlen im Ordner")
        return
    fake = types.ModuleType("sounddevice")
    fake.query_devices = lambda i=None: ({"name": "Fake", "hostapi": 0, "max_input_channels": 4} if i is not None
                                         else [{"name": "Fake", "hostapi": 0, "max_input_channels": 4}])
    fake.query_hostapis = lambda i=None: {"name": "FakeAPI"}
    fake.InputStream = FakeInputStream
    sys.modules["sounddevice"] = fake
    import live_anonymize
    rng = np.random.default_rng(3)
    audio = (0.05 * rng.standard_normal((10 * SR, 4))).astype(np.float32)
    FakeInputStream.audio = audio
    use = [3, 2, 1, 0]
    with tempfile.TemporaryDirectory() as d:
        out, raw = os.path.join(d, "out.wav"), os.path.join(d, "raw.wav")
        base = ["live_anonymize.py", "--dummy-sep", "--device", "cpu", "--input-device", "0", "--channels", "4",
                "--use-channels", ",".join(map(str, use)), "--seconds", "12", "--out", out]

        def run(extra):
            sys.argv = base + extra
            live_anonymize.main()

        orig_input = builtins.input
        builtins.input = lambda *a: "n"                       # Rueckfrage mit Nein beantworten
        try:
            run(["--raw-out", raw])
            raise AssertionError("SystemExit erwartet")
        except SystemExit:
            pass
        finally:
            builtins.input = orig_input
        assert not os.path.exists(raw)
        ok("--raw-out: Rueckfrage mit 'n' bricht ab, es entsteht keine Rohdatei")
        run([])
        assert not os.path.exists(raw)
        ok("ohne --raw-out: keine Rohdatei")
        run(["--raw-out", raw, "--yes"])
        r, fs = sf.read(raw, dtype="float32", always_2d=True)
        info = sf.info(raw)
        assert fs == SR and r.shape[1] == 4 and info.subtype == "FLOAT", (fs, r.shape, info.subtype)
        assert len(r) >= 8 * SR and np.array_equal(r, audio[:len(r)][:, use]), "Rohdatei != Eingang (Kanalzuordnung?)"
        o, _ = sf.read(out, always_2d=True)
        assert o.shape[1] == 1 and len(o) >= 2 * SR and len(o) <= len(r)
        ok(f"--raw-out --yes: Rohdatei 4 Kanaele, FLOAT, {len(r) / SR:.1f} s, bitgleich zum Eingang mit "
           f"Kanalzuordnung {use}; anonymisierte Ausgabe mono, {len(o) / SR:.1f} s")


TESTS = dict(api=test_api, eval=test_eval, vad=test_vad_helper, skip=test_skip_mono, mc=test_skip_mc, raw=test_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="optional: echter Checkpoint fuer den API-Test")
    ap.add_argument("--only", default=",".join(TESTS), help="kommagetrennt: " + ",".join(TESTS))
    args = ap.parse_args()
    for name in args.only.split(","):
        print(f"\n== {name} ==", flush=True)
        TESTS[name](args)
    print(f"\nALLE TESTS OK ({len(RESULTS)} Pruefungen)")


if __name__ == "__main__":
    main()