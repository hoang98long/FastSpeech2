#%%
import requests
import pytest
import json
from utils import read_data, url, log_request, headers

from typing import Any, Dict
import time
#%%

@pytest.mark.skip()
@log_request
def get(payloads)-> Any:
    payloads = list(map( json.dumps, payloads))
    # response = requests.request("GET", url, headers=headers, data=payload)
    responses = [ requests.request('GET', url, headers=headers, data=p) for p in payloads ]
    # return response.content
    return responses



def run():
    payloads = read_data()
    responses = get(payloads)



    
if __name__ == '__main__':
    payloads = read_data()
    n = len(payloads)
    _start = time.time()
    for payload in payloads:
        transfer(payload)
    
    _end = time.time()

