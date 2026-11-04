"""
Mehrkanal-Anonymisierung fuer einen Ordner (4-Kanal-WAV rein, 4-Kanal-WAV raus).

Nutzt NBSSSeparator aus nbss_separator.py direkt: separate(x[C, T]) -> (speech[T], background[T]),
geschaetzt am Referenzkanal 0 der Eingabe. Dateien > seg (20 s) werden vom Separator selbst
in Chunks mit Links-Kontext verarbeitet.

Beispiel:
  python anonymize_audio_folder_mc.py -i /project/brunke/nbss_runs/.../mixtures \
      -o /project/brunke/nbss_runs/.../anon_mc -ckpt <ckpt> --seed 0 --vad-mode ref
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from nbss_separator import NBSSSeparator
from mc_anonymizer import blur_audio_mc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beats"))
from BEATs import BEATs, BEATsConfig  # noqa: E402

SR = 16000


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and a.gpu_frac:  # GPU-Deckel (Training laeuft parallel); falls ihr einen eigenen Helfer habt, hier ersetzen
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    source_sep = NBSSSeparator(a.ckpt, device=str(device))

    ckpt = torch.load(os.path.abspath("./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
                      map_location=device)
    vad = BEATs(BEATsConfig(ckpt["cfg"]))
    vad.load_state_dict(ckpt["model"])
    vad.eval().to(device)
    with open("./utils/audioset_ontology.json") as f:
        name_to_id = {e["name"]: e["id"] for e in json.load(f)}
    speech_idx = next(k for k, v in ckpt["label_dict"].items() if v == name_to_id["Speech"])

    sep_ref0 = source_sep.separate

    os.makedirs(a.output_dir, exist_ok=True)
    wavs = [f for f in sorted(os.listdir(a.input_dir)) if f.lower().endswith(".wav")]
    for i, f in enumerate(wavs):
        y, sr = sf.read(os.path.join(a.input_dir, f), dtype="float32", always_2d=True)
        assert sr == SR, f"{f}: {sr} Hz"
        assert y.shape[1] == NBSSSeparator.N_CHN, f"{f}: {y.shape[1]} Kanaele"
        y = y.T  # [C, T]
        out, _ = blur_audio_mc(
            y, SR, int(SR * a.segmin), int(SR * a.segmax), int(SR * a.framemin), sep_ref0,
            seghopmin_perc=a.seghopmin, seghopmax_perc=a.seghopmax, framehopmin_perc=a.framehopmin,
            mixframe=a.mixframe, reverse=a.reverse, vad_model=vad, vad_sr=SR,
            vad_speech_index=speech_idx, vad_speech_ths=0.3, vad_mode=a.vad_mode,
            device=device, seed=a.seed + i, tail_mode=a.tail_mode,
        )
        sf.write(os.path.join(a.output_dir, f), out.T, SR)
        print("done", f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_dir", required=True)
    p.add_argument("-o", "--output_dir", required=True)
    p.add_argument("-ckpt", "--ckpt", required=True)
    p.add_argument("--gpu-frac", type=float, default=0.25, help="Anteil GPU-Speicher (0 = kein Deckel)")
    p.add_argument("--seed", type=int, default=0, help="Basisseed; Datei i bekommt seed+i (reproduzierbar)")
    p.add_argument("--vad-mode", choices=["ref", "any"], default="ref")
    p.add_argument("--tail-mode", choices=["last_window", "raw"], default="last_window")
    p.add_argument("-segmin", type=float, default=2.0)
    p.add_argument("-segmax", type=float, default=2.0)
    p.add_argument("-seghopmin", type=float, default=0.95)
    p.add_argument("-seghopmax", type=float, default=0.95)
    p.add_argument("-framemin", type=float, default=0.0)
    p.add_argument("-framehopmin", type=float, default=0.95)
    p.add_argument("-mixframe", type=str2bool, default=False)
    p.add_argument("-reverse", choices=["backward", "forward", "random"], default="backward")
    main(p.parse_args())
