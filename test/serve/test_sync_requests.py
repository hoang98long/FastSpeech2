#%%
import requests
import json
from utils import read_data, url, log_request

from typing import Any, Dict
import time
#%%

headers = {
  'Content-Type': 'application/json'
}

def transfer(payload )-> Any:
    payload = json.dumps(payload)
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.content

    
if __name__ == '__main__':
    payloads = read_data()
    n = len(payloads)
    _start = time.time()
    for payload in payloads:
        transfer(payload)
    
    _end = time.time()

