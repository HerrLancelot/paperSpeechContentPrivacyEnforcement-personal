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

    def __init__(self, ckpt_path, device="cpu", use_bf16=True, seg=20.0, ctx=2.0):
        """
        Args:
            ckpt_path: Lightning checkpoint (.ckpt) of the training run
            use_bf16:  run the network under bf16 autocast like the training (precision: bf16-mixed)
            seg:       chunk length in s for long files (None = whole file at once)
            ctx:       left context in s that is processed but discarded (model is causal)
        """
        self.device = torch.device(device)
        self.sr = self.SR
        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        self.seg, self.ctx = seg, ctx

        self.arch = OnlineSpatialNet(
            dim_input=2 * self.N_CHN, dim_output=4, num_layers=8, dim_squeeze=8, num_freqs=257,
            encoder_kernel_size=5, dim_hidden=96, dim_ffn=192, num_heads=4,
            dropout=(0.0, 0.0, 0.0), kernel_size=(5, 3), conv_groups=(8, 8),
            norms=["LN", "LN", "GN", "LN", "LN", "LN"], padding="zeros", full_share=0,
            attention="ret(2)", decay=[4, 5, 9, 10], chunkwise_recurrent=True, rope=False,
        )
        self.stft = STFT(n_fft=512, n_hop=256, win_len=512, win="hann_window")
        self.norm = Norm(mode="utterance", online=True)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {}
        for k, v in ckpt["state_dict"].items():
            k = k.replace("_orig_mod.", "")          # training used compile: true
            if k.startswith("arch."):
                sd[k[len("arch."):]] = v
        self.arch.load_state_dict(sd, strict=True)   # raises if config and checkpoint differ

        for m in (self.arch, self.stft, self.norm):
            m.eval().to(self.device)

    @torch.no_grad()
    def _forward(self, x):
        """x: [B, C, T] -> [B, Spk, T]; same steps as TrainModule.forward (loss neg_snr, no mask)."""
        X, stft_paras = self.stft.stft(x)                                   # [B,C,F,T] complex
        B, C, F, T = X.shape
        X, (Xr, XrMM) = self.norm.norm(X, ref_channel=self.REF_CHANNEL)
        X = torch.view_as_real(X.permute(0, 2, 3, 1)).reshape(B, F, T, -1)  # [B,F,T,2C]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.arch(X)
        out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2)).permute(0, 3, 1, 2)  # [B,Spk,F,T]
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
        sep.use_bf16, sep.seg = True, None
        s_bf16, _ = sep.separate(x)
        print(f"fp32 vs bf16 (speech): {rel_db(s_whole, s_bf16):.1f} dB")
    print(f"rms in(ref)={np.sqrt((x[0]**2).mean()):.4f} speech={np.sqrt((s_whole**2).mean()):.4f} "
          f"bg={np.sqrt((b_whole**2).mean()):.4f}")
    sf.write(os.path.join(out_dir, "speech.wav"), s_whole, sep.sr)
    sf.write(os.path.join(out_dir, "background.wav"), b_whole, sep.sr)