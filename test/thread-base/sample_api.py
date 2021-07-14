from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from utils import msg_client as mc
import uvicorn
import random

from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
app = FastAPI()

origins = [
        '*'
        ]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NUM_MODEL = 5
model_ids = [0]*NUM_MODEL

class Item(BaseModel):
    text: str

@app.get('/healthcheck')
def healthcheck():
    print('Send and request with queue')
    queue_size = mc.send_and_get(model='queue')
    return {"queue_size": queue_size}


@app.post("/tts/generate")
async def root(request: Request, item: Item):
    text = item.text
    i = random.randint(0, 4)
    model_name = f"model-{i:02d}"
    # wav_file = e2e(text)
    

    return FileResponse(wav_file)


@app.route('/model1')
def model1():
    results = mc.send_and_get(model='model1')
    return {"results": results}

# if __name__ == '__main__':
#     mc.start()
    # app.run(host='0.0.0.0')
    # uvicorn.run(app, host='0.0.0.0')


if __name__ == '__main__':
    mc.start()
    uvicorn.run(app, port=80, host='0.0.0.0', debug=True)
