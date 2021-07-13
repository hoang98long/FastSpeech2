import argparse
from logging import debug

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from synthesize import preprocess_english, preprocess_mandarin, synthesize_wav
from utils.model import get_model, get_vocoder
from utils.tools import synth_samples, synth_wav, to_device

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from fastapi.templating import Jinja2Templates
origins = [
        '*'
        ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read Config
preprocess_config = yaml.load(
    open( './config/Viet_tts/preprocess.yaml', "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(open('./config/Viet_tts/model.yaml', "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open('./config/Viet_tts/train.yaml', "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)
templates = Jinja2Templates(directory="templates")
# Get model

class Args:
    restore_step = 3000

args = Args()

class Item(BaseModel):
    text: str

model = get_model(args, configs, device, train=False)

# Load vocoder
vocoder = get_vocoder(model_config, device)
restore_step = 5000
control_values = 1., 1., 1.
@app.get("/tts/generate")
async def root(request:Request, item: Item):
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
    # return FileResponse(wav_file)
    return templates.TemplateResponse("home.html", {"request": request})
    # wav_stream = open(wav_file, mode='rb')
    # return StreamingResponse(wav_stream, media_type="video/mp4")
	# return {"message": "Hello World"}
if __name__ == '__main__':
    uvicorn.run(app, port=80, host='0.0.0.0', debug=True)
