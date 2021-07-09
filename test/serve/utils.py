#%%
import os
from typing import List, Any
import requests
import time
#%%
url = "http://183.91.2.4:4097/tts/generate"
try:
    cur_dir = os.path.dirname(__file__)
except:
    cur_dir = '.'

def read_data(fpath: str = os.path.join(cur_dir, './data.txt')):
    assert os.path.isfile(fpath)
    with open(fpath, 'r') as fr:
        data = fr.read().strip().split('\n')

    return data

#TODO: print -> logging
def log_request(type:str = 'sync'):
    if type == 'sync':
        get_time = time.time
    def wrapper(f, payloads: List[Any], *args, **kwargs):

        _start = get_time()
        response = f(payloads)
        _end = get_time()
        avg_time = (_end-_start)/len(payloads)

        return response
    return wrapper
