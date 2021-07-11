import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from fastapi import FastAPI

import uvicorn
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import queue
# from calc import cpu_bound_func
# Model preparing

import torch
import yaml
import numpy as np

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, synth_wav

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
    restore_step = 3000

args = Args()

class Item(BaseModel):
    text: str

# Load vocoder
n_instance = 5
vocoders = [get_vocoder(model_config, device) for i in range(n_instance)]
restore_step = 5000
control_values = 1., 1., 1.

models = [get_model(args, configs, device, train=False) for i in range(n_instance)]
model_queue = queue.Queue()

for m, v in zip(models, vocoders):
    model_queue.put((m, v))

app = FastAPI()


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(app.state.executor, fn, *args)  # wait and return result

async def run_models(batchs):
    # TODO: check empy
    model, vocoder = model_queue.get()
    wav_files = await run_in_process(synthesize_wav, model, restore_step, configs, vocoder, batchs, control_values)
    model_queue.put((model, vocoder))
    return wav_files[0]

@app.get("/tts/generate")
async def root(item: Item):
    text = item.text
    ids = raw_texts = text
    speakers = np.array([0])
    texts = np.array([preprocess_english(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    # for wav_file in synthesize_wav(model, restore_step, configs, vocoder, batchs, control_values):
    #     break
    wav_file = await run_models(batchs)
    return FileResponse(wav_file)
    # res = await run_in_process(cpu_bound_func, param)
    # return {"result": res}


@app.on_event("startup")
async def on_startup():
    app.state.executor = ProcessPoolExecutor()


@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()


if __name__ == '__main__':
    uvicorn.run(app, port=80, host='0.0.0.0')
