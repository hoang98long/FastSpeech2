#%%
import requests
import pytest
import json
from utils import read_data, url, log_request, headers
import random
from typing import Any, Dict
import time
#%%

# @pytest.mark.skip()
@log_request()
def get(payloads)-> Any:
    payloads = list(map( lambda x: json.dumps({'text': x}), payloads))
    # print(f"Reading data")
    print(payloads[0])
    responses = [ requests.request('GET', url, headers=headers, data=p) for p in payloads ]
    # return response.content
    return responses



def test_request():
    # print('Inside test sync request')
    # import pdb; pdb.set_tra()
    payloads = read_data()
    random.shuffle(payloads)
    for _ in range(10):
        responses = get(payloads)



    
if __name__ == '__main__':
    pass
    # payloads = read_data()
    # n = len(payloads)
    # _start = time.time()
    # for payload in payloads:
    #     test(payload)
    
    _end = time.time()

