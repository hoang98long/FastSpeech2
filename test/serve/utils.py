#%%
import os
from typing import List, Any, Callable
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)
#%%
headers = {
  'Content-Type': 'application/json'
}
# url = "http://183.91.2.4:4097/tts/generate"
url = "http://127.0.0.1:80/tts/generate"
try:
    cur_dir = os.path.dirname(__file__)
except:
    cur_dir = '.'

def read_data(fpath: str = os.path.join(cur_dir, './data.txt')):
    assert os.path.isfile(fpath)
    with open(fpath, 'r') as fr:
        data = fr.read().strip().split('\n')
    return data[:50]

def log_request(is_async=False):
    if not is_async:
        get_time = time.time
    else :
        get_time = time.perf_counter

    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(payloads: List[Any], *args, **kwargs):
            _start_mess = f'Requesting {len(payloads)} payloads'
            logger.info(_start_mess)
            _start = get_time()
            responses = func(payloads)
            _end = get_time()
            avg_time = (_end-_start)/len(payloads)
            _end_mess = f"Request done with {avg_time}s/payload"
            # print(_end_mess)
            logger.info(_end_mess)
            return responses

        @wraps(func)
        async def async_wrapper(payloads: List[Any], *args, **kwargs):
            _start_mess = f'Requesting {len(payloads)} payloads'
            logger.info(_start_mess)
            print(_start_mess)
            _start = get_time()
            responses = await func(payloads)
            _end = get_time()
            avg_time = (_end-_start)/len(payloads)
            _end_mess = f"Request done with {avg_time}s/payload"
            # print(_end_mess)
            logger.info(_end_mess)
            print(_end_mess)
            return responses
        wrapper = async_wrapper if is_async else sync_wrapper
        return wrapper
    return decorator
