"""OnlineSpatialNet (NBSS) as drop-in source separator for the Tailleur et al. pipeline.

Expects the NBSS code (folders models/arch and models/io) in ./nbss/models next to this file.
All model settings below are copied 1:1 from the config.yaml of the training run.

Model I/O (no PIT in training -> fixed output order, see SPEECH_IDX / BG_IDX):
    input : [C=4, T] float waveform at 16 kHz (mic order as in training, reference channel 0)
    output: speech estimate and background estimate, both [T]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbss"))
from models.arch.OnlineSpatialNet import OnlineSpatialNet  # noqa: E402
from models.io.norm import Norm  # noqa: E402
from models.io.stft import STFT  # noqa: E402


class NBSSSeparator:
    SR = 16000
    N_CHN = 4
    REF_CHANNEL = 0       # index inside the 4 channels (config: ref_channel: 0, channels: [0,1,2,3])
    # TODO: verify against the target order in data_loaders/my_dataset.py (ys: [B, Spk, C, T])
    SPEECH_IDX = 0
    BG_IDX = 1

    # attention settings of the two trained variants (everything else in the config is identical)
    ATTENTION = {"retention": "ret(2)", "mamba": "mamba(16,4)"}
    # Rechengenauigkeit des Netzes (Autocast); STFT/Norm/iSTFT laufen immer in fp32
    AMP_DTYPE = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}

    def __init__(self, ckpt_path, device="cpu", use_bf16=True, seg=20.0, ctx=2.0, arch="retention",
                 precision=None):
        """
        Args:
            ckpt_path: Lightning checkpoint (.ckpt) of the training run
            arch:      "retention" (attention ret(2)) or "mamba" (attention mamba(16,4), needs mamba_ssm, Linux)
            use_bf16:  run the network under bf16 autocast like the training (precision: bf16-mixed)
            precision: "fp32" | "bf16" | "fp16" (Autocast, nur auf CUDA); None = aus use_bf16 abgeleitet.
                       fp16: kleinerer Wertebereich als bf16; bei nicht-endlichen Werten wird der Aufruf
                       in fp32 wiederholt und in self.n_fp16_fallback gezaehlt.
            seg:       chunk length in s for long files (None = whole file at once)
            ctx:       left context in s that is processed but discarded (model is causal)
        """
        self.device = torch.device(device)
        self.sr = self.SR
        if precision is None:
            precision = "bf16" if use_bf16 else "fp32"
        if precision not in self.AMP_DTYPE:
            raise ValueError(f"precision muss fp32, bf16 oder fp16 sein, nicht '{precision}'")
        self.precision = precision if self.device.type == "cuda" else "fp32"   # CPU: nur fp32
        self.n_fp16_fallback = 0
        self.seg, self.ctx = seg, ctx

        self.arch = OnlineSpatialNet(
            dim_input=2 * self.N_CHN, dim_output=4, num_layers=8, dim_squeeze=8, num_freqs=257,
            encoder_kernel_size=5, dim_hidden=96, dim_ffn=192, num_heads=4,
            dropout=(0.0, 0.0, 0.0), kernel_size=(5, 3), conv_groups=(8, 8),
            norms=["LN", "LN", "GN", "LN", "LN", "LN"], padding="zeros", full_share=0,
            attention=self.ATTENTION[arch], decay=[4, 5, 9, 10], chunkwise_recurrent=True, rope=False,
        )
        self.stft = STFT(n_fft=512, n_hop=256, win_len=512, win="hann_window")
        self.norm = Norm(mode="utterance", online=True)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {}
        for k, v in ckpt["state_dict"].items():
            k = k.replace("_orig_mod.", "")          # training used compile: true
            if k.startswith("arch."):
                sd[k[len("arch."):]] = v
        try:
            self.arch.load_state_dict(sd, strict=True)   # raises if config and checkpoint differ
        except RuntimeError as e:
            raise RuntimeError(f"Checkpoint passt nicht zu arch='{arch}' (retention/mamba verwechselt?): {e}") from e

        for m in (self.arch, self.stft, self.norm):
            m.eval().to(self.device)

    @property
    def use_bf16(self):                      # Kompatibilitaet zu aelteren Skripten
        return self.precision == "bf16"

    @use_bf16.setter
    def use_bf16(self, v):
        self.precision = "bf16" if v else "fp32"

    @torch.no_grad()
    def _forward(self, x):
        """x: [B, C, T] -> [B, Spk, T]; same steps as TrainModule.forward (loss neg_snr, no mask)."""
        X, stft_paras = self.stft.stft(x)                                   # [B,C,F,T] complex
        B, C, F, T = X.shape
        X, (Xr, XrMM) = self.norm.norm(X, ref_channel=self.REF_CHANNEL)
        X = torch.view_as_real(X.permute(0, 2, 3, 1)).reshape(B, F, T, -1)  # [B,F,T,2C]
        amp = self.AMP_DTYPE[self.precision] if self.device.type == "cuda" else None
        with torch.autocast(device_type=self.device.type, dtype=amp or torch.bfloat16, enabled=amp is not None):
            out = self.arch(X)
        out = out.float()
        if self.precision == "fp16" and amp is not None and not torch.isfinite(out).all():
            self.n_fp16_fallback += 1               # Ueberlauf in fp16 -> dieser Aufruf in fp32
            out = self.arch(X).float()
        out = torch.view_as_complex(out.reshape(B, F, T, -1, 2)).permute(0, 3, 1, 2)  # [B,Spk,F,T]
        out = self.norm.inorm(out, (Xr, XrMM))
        return self.stft.istft(out, stft_paras)                             # [B,Spk,T]

    @torch.no_grad()
    def separate(self, x):
        """x: numpy [C, T] at 16 kHz -> (y_speech [T], y_background [T]) as float32 numpy."""
        if x.ndim != 2 or x.shape[0] != self.N_CHN:
            raise ValueError(f"NBSSSeparator expects [{self.N_CHN}, T], got {tuple(x.shape)}")
        x = torch.from_numpy(np.ascontiguousarray(x)).float().to(self.device)
        T = x.shape[-1]
        if self.seg is None or T <= int(self.seg * self.sr):
            y = self._forward(x[None])[0]
        else:
            L, c = int(self.seg * self.sr), int(self.ctx * self.sr)
            outs = []
            for s in range(0, T, L):                 # causal model: left context only
                a, e = max(0, s - c), min(T, s + L)
                outs.append(self._forward(x[None, :, a:e])[0][:, s - a:])
            y = torch.cat(outs, dim=-1)
        y = y.cpu().numpy()
        return y[self.SPEECH_IDX], y[self.BG_IDX]

    @torch.no_grad()
    def separate_batch(self, xs):
        """xs: numpy [B, C, T] at 16 kHz -> (speech [B, T], background [B, T]) float32 numpy.
        Same chunking as separate(), but all B inputs run through the network together."""
        if xs.ndim != 3 or xs.shape[1] != self.N_CHN:
            raise ValueError(f"expects [B, {self.N_CHN}, T], got {tuple(xs.shape)}")
        x = torch.from_numpy(np.ascontiguousarray(xs)).float().to(self.device)
        T = x.shape[-1]
        if self.seg is None or T <= int(self.seg * self.sr):
            y = self._forward(x)
        else:
            L, c = int(self.seg * self.sr), int(self.ctx * self.sr)
            outs = []
            for s in range(0, T, L):
                a, e = max(0, s - c), min(T, s + L)
                outs.append(self._forward(x[:, :, a:e])[:, :, s - a:])
            y = torch.cat(outs, dim=-1)
        y = y.cpu().numpy()
        return y[:, self.SPEECH_IDX], y[:, self.BG_IDX]


if __name__ == "__main__":
    # Self-test: python nbss_separator.py CKPT 4ch_16k.wav OUT_DIR
    import soundfile as sf
    ckpt, wav, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    assert sr == NBSSSeparator.SR and x.shape[1] == NBSSSeparator.N_CHN, (sr, x.shape)
    x = x.T
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def rel_db(a, b):  # how close b is to a (higher = closer)
        return 10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-12))

    sep = NBSSSeparator(ckpt, dev, use_bf16=False, seg=None)
    s_whole, b_whole = sep.separate(x)
    sep.seg = 10.0
    s_chunk, _ = sep.separate(x)
    print(f"whole vs chunked (speech): {rel_db(s_whole, s_chunk):.1f} dB")
    if dev == "cuda":
        for prec in ("bf16", "fp16"):
            sep.precision, sep.seg = prec, None
            s_p, _ = sep.separate(x)
            print(f"fp32 vs {prec} (speech): {rel_db(s_whole, s_p):.1f} dB"
                  f"{'  (fp16-Fallbacks: %d)' % sep.n_fp16_fallback if prec == 'fp16' else ''}")
        sep.precision = "fp32"                  # Rest des Selbsttests wieder in fp32
    print(f"rms in(ref)={np.sqrt((x[0]**2).mean()):.4f} speech={np.sqrt((s_whole**2).mean()):.4f} "
          f"bg={np.sqrt((b_whole**2).mean()):.4f}")
    # batch vs. sequential over the 4 rolled channels (checks that Norm treats batch items independently)
    rolls = np.stack([np.roll(x, -c, axis=0) for c in range(4)])
    sb, _ = sep.separate_batch(rolls)
    seq = np.stack([sep.separate(r)[0] for r in rolls])
    print("batch vs sequential (rolled, speech), dB per roll:", [round(rel_db(seq[i], sb[i]), 1) for i in range(4)])
    sf.write(os.path.join(out_dir, "speech.wav"), s_whole, sep.sr)
    sf.write(os.path.join(out_dir, "background.wav"), b_whole, sep.sr)