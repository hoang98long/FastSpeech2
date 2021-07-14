import os
from types import SimpleNamespace
import zmq.green as zmq
from utils import pickle
from utils import shared_memory as sm
import torch.multiprocessing as mp
import threading
import time

QUEUE_SIZE = mp.Value('i', 0)
TOPIC = 'snaptravel'
prediction_functions = {}
RECEIVE_PORT = os.getenv("RECEIVE_PORT")
SEND_PORT = os.getenv("SEND_PORT")
NUM_MODEL = 5

import yaml
# Read Config
pre_path = '../../'
preprocess_config = yaml.load(
    open(os.path.join(pre_path,  './config/Viet_tts/preprocess.yaml'), "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(open(os.path.join(pre_path, './config/Viet_tts/model.yaml'), "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(os.path.join(pre_path, './config/Viet_tts/train.yaml'), "r"), Loader=yaml.FullLoader)
from e2e_model import E2E

class Args:
    restore_step = 50000

args = Args()
restore_step = args.restore_step

def _parse_recv_for_json(result, topic=TOPIC):
  print(f"Inside parse json, {result}")
  compressed_json = result[len(topic) + 1:]
  return pickle.decompress(compressed_json)

def _decrease_queue():
  with QUEUE_SIZE.get_lock():
    QUEUE_SIZE.value -= 1

def _increase_queue():
  with QUEUE_SIZE.get_lock():
    QUEUE_SIZE.value += 1
    
def send_prediction(message, result_publisher, topic=TOPIC):
  _increase_queue()

  print('Worker send prediction')
  model_name = message['model']
  body = message['body']
  id = message['id']

  if not model_name:
    compressed_message = pickle.compress({'error': True, 'error_msg': 'Model doesn\'t exist', 'id': id})
    result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
    _decrease_queue()
    return

  predict = prediction_functions.get(model_name)
  f = sm.function_wrapper(predict)
  time.sleep(2.)
  result = sm.run_function(f, *body)

  if result.get('error'):
    compressed_message = pickle.compress({'error': True, 'error_msg': result['error'], 'id': id})
    result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
    _decrease_queue()
    return

  if result.get('result') is None:
    compressed_message = pickle.compress({'error': True, 'error_msg': 'No result was given: ' + str(result), 'id': id})
    result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
    _decrease_queue()
    return

  prediction = result['result']

  compressed_message = pickle.compress({'prediction': prediction, 'id': id})
  result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
  _decrease_queue()

def queue_size():
  return QUEUE_SIZE.value

def load_models():
  models = SimpleNamespace()
  # This is where you load your models
  # For example, model.model1 = model1.load_model()
  # where
  # `model1.py` has 
  # def load_model():
  #   archive = load_archive(SERIALIZATION_DIR)
  #   archive.model.share_memory()
  #   predictor = Predictor.from_archive(archive, 'model')
  #   return predictor
  models = {f'model-{i:02d}': E2E(args, preprocess_config, model_config, train_config) for i in range(NUM_MODEL)}
  return models

def start():
  print('Worker started')
  global prediction_functions
  
  models = load_models()
  
  # prediction_functions = {
  #   # This is where you would add your models for inference
  #   # For example, 'model1': model.model1.predict,
  #   #              'model2': model.model2.predict,
  #   'queue': queue_size
  # }

  prediction_functions = {
          f"model-{i:02d}": models[f"model-{i:02d}"].forward for i in range(NUM_MODEL)
          }
  prediction_functions['queue'] = queue_size

  print(f'Connecting to {RECEIVE_PORT} in server')
  context = zmq.Context()
  work_subscriber = context.socket(zmq.SUB)
  work_subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC.encode('utf8'))
  work_subscriber.bind(f'tcp://127.0.0.1:{RECEIVE_PORT}')

  # send work
  print(f'Connecting to {SEND_PORT} in server')
  result_publisher = context.socket(zmq.PUB)
  result_publisher.bind(f'tcp://127.0.0.1:{SEND_PORT}')

  print('Server started')
  while True:
    message = _parse_recv_for_json(work_subscriber.recv())
    threading.Thread(target=send_prediction, args=(message, result_publisher), kwargs={'topic': TOPIC}).start()

if __name__ == '__main__':
  start()
