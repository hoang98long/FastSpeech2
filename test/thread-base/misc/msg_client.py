import json
import uuid

import zmq.green as zmq # required since we are in gevents

from utils import pickle
import os

work_publisher = None
result_subscriber = None
TOPIC = 'snaptravel'

SEND_PORT = os.getenv("CLIENT_SEND_PORT")
RECEIVE_PORT = os.getenv("CLIENT_RECEIVE_PORT")


def start():
  global work_publisher, result_subscriber

  context = zmq.Context()
  work_publisher = context.socket(zmq.PUB)
  work_publisher.connect(f'tcp://127.0.0.1:{SEND_PORT}')
  print(f'client_publisher connected to {SEND_PORT}')

def _parse_recv_for_json(result, topic=TOPIC):
  compressed_json = result[len(topic) + 1:]
  return pickle.decompress(compressed_json)


def send(*args, model=None, topic=TOPIC):
  print('Client send reuqest')
  id = str(uuid.uuid4())
  message = {'body': args, 'model': model, 'id': id}
  compressed_message = pickle.compress(message)
  work_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
  return id


def get(id, topic=TOPIC):
  print('Client get request')
  context = zmq.Context()
  result_subscriber = context.socket(zmq.SUB)
  result_subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode('utf8'))
  result_subscriber.connect(f'tcp://127.0.0.1:{RECEIVE_PORT}')
  print(f"Client_subscriber connected to {RECEIVE_PORT}")
  result = _parse_recv_for_json(result_subscriber.recv())
  while result['id'] != id:
    result = _parse_recv_for_json(result_subscriber.recv())

  result_subscriber.close()

  if result.get('error'):
    raise Exception(result['error_msg'])

  return result['prediction']


def send_and_get(*args, model=None):
  id = send(*args, model=model)
  return get(id)
