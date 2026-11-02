"""Streaming-Bausteine fuer OnlineSpatialNet (Retention-Variante, ret(2), rope=False).

Dieselben Gewichte, nur anderer Rechenweg: Frame fuer Frame mit Zustand ueber Aufrufe hinweg.
Das Repo selbst haelt keinen Zustand zwischen forward()-Aufrufen, deshalb liegt er hier:

  * ArchStreamer : Encoder-Konv-Zustand, pro Layer ein Dict (Retention-Zustand + CausalConv1d-Zustaende)
  * StreamSTFT   : STFT-Ueberlapp, identisch zu torch.stft(center=True, reflect)
  * StreamISTFT  : Overlap-Add inkl. Fenster-Huellkurve, identisch zu torch.istft(center=True)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch import OnlineSpatialNet as OSN


# --------------------------------------------------------------------------------------
# Fix fuer CausalConv1d: im Repo steht `x[..., -self.kernel_size + 1:]`. kernel_size ist ein
# Tuple, `-tuple` wirft einen TypeError. Der Pfad wird im Repo nie benutzt (state ist immer None).
# --------------------------------------------------------------------------------------
def _causal_forward(self, x, state=None):
    k1 = self.kernel_size[0] - 1
    if state is None or id(self) not in state:
        x = F.pad(x, pad=(k1 - self.look_ahead, self.look_ahead))
    else:
        x = torch.cat([state[id(self)], x], dim=-1)
    if state is not None:
        state[id(self)] = x[..., x.shape[-1] - k1:]
    return nn.Conv1d.forward(self, x)


def patch_causal_conv():
    import inspect
    src = inspect.getsource(OSN.CausalConv1d.forward)
    if "F.pad" not in src or "mode" in src:
        print("[!] CausalConv1d.forward im Repo weicht ab (Padding-Modus?) -> _causal_forward pruefen:\n" + src)
    OSN.CausalConv1d.forward = _causal_forward


class ArchStreamer:
    """Ein Frame pro Aufruf: x [B, F, 1, H0] -> [B, F, 1, dim_out]."""

    def __init__(self, arch: OSN.OnlineSpatialNet):
        assert arch.pos is not None, "nur Retention ('ret(...)') wird unterstuetzt"
        patch_causal_conv()
        self.arch = arch
        self.reset()

    def reset(self):
        self.t = 0
        self.enc_state = {}
        self.layer_states = [dict() for _ in self.arch.layers]

    def step(self, x: torch.Tensor) -> torch.Tensor:
        a = self.arch
        B, Fq, T, H0 = x.shape
        assert T == 1, "genau ein Frame pro Aufruf"
        h = a.encoder(x.reshape(B * Fq, T, H0).permute(0, 2, 1), state=self.enc_state).permute(0, 2, 1)
        h = h.reshape(B, Fq, T, -1)
        rel_pos = a.pos.forward(slen=self.t, activate_recurrent=True)
        for layer, st in zip(a.layers, self.layer_states):
            h, _ = layer(h, rel_pos, False, a.rope, st, False)
        self.t += 1
        return a.decoder(h)


class StreamSTFT:
    """Frames wie torch.stft(center=True, pad_mode='reflect'). push(x[C,n]) -> [C,F,k] oder None."""

    def __init__(self, stft):
        self.n_fft, self.hop = stft.n_fft, stft.n_hop
        assert stft.win_len == self.n_fft
        self.window = stft.window
        self.pad = self.n_fft // 2
        self.pending = None
        self.buf = None

    def push(self, x: torch.Tensor):
        if self.buf is None:
            self.pending = x if self.pending is None else torch.cat([self.pending, x], dim=-1)
            if self.pending.shape[-1] < self.pad + 1:  # Reflect-Padding braucht x[pad]
                return None
            left = self.pending[..., 1:self.pad + 1].flip(-1)
            self.buf = torch.cat([left, self.pending], dim=-1)
            self.pending = None
        else:
            self.buf = torch.cat([self.buf, x], dim=-1)
        frames = []
        while self.buf.shape[-1] >= self.n_fft:
            frames.append(torch.fft.rfft(self.buf[..., :self.n_fft] * self.window, dim=-1))
            self.buf = self.buf[..., self.hop:]
        return torch.stack(frames, dim=-1) if frames else None


class StreamISTFT:
    """Overlap-Add wie torch.istft(center=True). push(Y[S,F]) -> [S,hop] (erster Frame: None).

    Ausgabe-Latenz: Frame t liefert die Samples [(t-1)*hop, t*hop) des Originals.
    """

    def __init__(self, stft):
        self.n_fft, self.hop = stft.n_fft, stft.n_hop
        assert self.n_fft == 2 * self.hop, "nur 50 % Ueberlapp"
        self.window = stft.window
        self.env = self.window[:self.hop] ** 2 + self.window[self.hop:] ** 2
        self.ola = None
        self.first = True

    def push(self, Y: torch.Tensor):
        y = torch.fft.irfft(Y, n=self.n_fft, dim=-1) * self.window
        if self.ola is None:
            self.ola = torch.zeros_like(y)
        self.ola = self.ola + y
        out = self.ola[..., :self.hop] / self.env
        self.ola = torch.cat([self.ola[..., self.hop:], torch.zeros_like(self.ola[..., :self.hop])], dim=-1)
        if self.first:  # Bereich vor Sample 0 (center=True), wird verworfen
            self.first = False
            return None
        return out