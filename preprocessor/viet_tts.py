import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    # speaker = "mta0"
    speakers = open(f"{in_dir}/speakers.txt", 'r').read().strip().split('\n')
    with open(os.path.join(in_dir, "meta_data.tsv"), encoding="utf-8") as f:
        texts = {}
        data = f.read().strip().split('\n')
        for i in range(len(data)):
            line = data[i]
            parts = line.strip().split("\t")
            base_name = os.path.splitext(os.path.basename(parts[0]))[0]
            text = parts[1]
            text = _clean_text(text, cleaners)
            # texts[i] = {base_name: text}
            texts[base_name] = text

    for speaker in speakers:
        wav_files = os.listdir(os.path.join(in_dir, speaker))
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        for wav_file in tqdm(wav_files):
            try:
                base_name = os.path.splitext(wav_file)[0]
                text = texts[base_name]
                wav_path = os.path.join(in_dir, speaker, wav_file)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
            except Exception as e:
                print(e)
                continue
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)
