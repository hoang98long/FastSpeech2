from torch import nn
import torch
# import yaml
from utils.model import get_model, get_vocoder
import numpy as np
from synthesize import preprocess_english
from utils.tools import to_device, synth_wav

control_values = 1., 1., 1.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class E2E(nn.Module):
    def __init__(self,args, preprocess_config, model_config, train_config):
        super(E2E, self).__init__()
        self.preprocess_config = preprocess_config
        configs = (preprocess_config, model_config, train_config)
        self.model = get_model(args, configs, device, train=False)
        # Load vocoder
        self.vocoder = get_vocoder(model_config, device)
        self.train_config = train_config
        self.model_config = model_config

    def forward(self, text):
        pitch_control, energy_control, duration_control = control_values
        ids = raw_texts = text
        texts = np.array([preprocess_english(text, self.preprocess_config)])
        speakers = np.array([0])
        text_lens = np.array([len(texts[0])])
        batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))

        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = self.model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            wav_file = synth_wav(
                batch,
                output,
                self.vocoder,
                self.model_config,
                self.preprocess_config,
                self.train_config["path"]["result_path"],
                )
    # print(f"Reference done after {time.time()-_start}")
        return wav_file
