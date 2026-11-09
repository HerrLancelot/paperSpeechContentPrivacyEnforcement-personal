"""
Zerlegt Mehrkanal-WAVs in Mono-Ordner: <out>/ch0 ... ch{C-1}, gleiche Dateinamen.
Damit lassen sich eval_wer_paper.py / eval_utility.py unveraendert pro Kanal ausfuehren.

  python split_channels.py -i <anon_mc_dir> -o <anon_mc_dir>_split
"""
import argparse
import os

import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("-i", "--input_dir", required=True)
p.add_argument("-o", "--output_dir", required=True)
a = p.parse_args()

for f in sorted(os.listdir(a.input_dir)):
    if not f.lower().endswith(".wav"):
        continue
    x, sr = sf.read(os.path.join(a.input_dir, f), dtype="float32", always_2d=True)
    for c in range(x.shape[1]):
        d = os.path.join(a.output_dir, f"ch{c}")
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, f), x[:, c], sr)
print("done")