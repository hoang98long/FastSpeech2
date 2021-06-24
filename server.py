from fastapi import FastAPI
import uvicorn
app = FastAPI()
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, synth_wav
from dataset import TextDataset
from text import text_to_sequence
import time

from synthesize import preprocess_english, preprocess_mandarin, synthesize_wav
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read Config
preprocess_config = yaml.load(
    open( './config/Viet_tts/preprocess.yaml', "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(open('./config/Viet_tts/model.yaml', "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open('./config/Viet_tts/train.yaml', "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

# Get model

class Args:
    restore_step = 5000
args = Args()

class Item(BaseModel):
    text: str

model = get_model(args, configs, device, train=False)

# Load vocoder
vocoder = get_vocoder(model_config, device)
restore_step = 5000
control_values = 1., 1., 1.
@app.get("/tts/generate")
async def root(item: Item):
    text = item.text
    ids = raw_texts = text
    speakers = np.array([0])
    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([preprocess_english(text, preprocess_config)])
    elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        texts = np.array([preprocess_mandarin(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    for wav_file in synthesize_wav(model, restore_step, configs, vocoder, batchs, control_values):
        break
    return FileResponse(wav_file)
    # wav_stream = open(wav_file, mode='rb')
    # return StreamingResponse(wav_stream, media_type="video/mp4")
	# return {"message": "Hello World"}
if __name__ == '__main__':
    uvicorn.run(app, port=80, host='0.0.0.0')
