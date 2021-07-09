#%%
import os
from typing import List, Any
import requests
import time
#%%
headers = {
  'Content-Type': 'application/json'
}
# url = "http://183.91.2.4:4097/tts/generate"
url = "http://0.0.0.0:80/tts/generate"
try:
    cur_dir = os.path.dirname(__file__)
except:
    cur_dir = '.'

def read_data(fpath: str = os.path.join(cur_dir, './data.txt')):
    assert os.path.isfile(fpath)
    with open(fpath, 'r') as fr:
        data = fr.read().strip().split('\n')

    return data[:10]

#TODO: print -> logging
def log_request(type:str = 'sync'):
    # print('Inside log_request')
    if type == 'sync':
        get_time = time.time
    def decorator(func: Any):
        # print('Decorator')
        def wrapper(payloads: List[Any], *args, **kwargs):
            print('Requesting {len(payloads)} payloads')
            _start = get_time()
            responses= func(payloads)
            _end = get_time()
            avg_time = (_end-_start)/len(payloads)
            print(f"Request done with {avg_time}s/payload")
            return responses
        return wrapper
    return decorator
