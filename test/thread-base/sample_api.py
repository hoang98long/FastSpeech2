from fastapi import FastAPI
from utils import msg_client as mc
import uvicorn

app = FastAPI()

# @app.route('/healthcheck')
@app.get('/healthcheck')
def healthcheck():
    print('Send and request with queue')
    queue_size = mc.send_and_get(model='queue')
    return {"queue_size": queue_size}

# @app.route('/model1')
@app.route('/model1')
def model1():
    results = mc.send_and_get(model='model1')
    return {"results": results}

if __name__ == '__main__':
    mc.start()
    # app.run(host='0.0.0.0')
    uvicorn.run(app, host='0.0.0.0')
