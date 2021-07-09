#%%
import requests
import pytest
import json
from utils import read_data, url, log_request, headers
import random
from typing import Any, Dict
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('sync_request.log')

logger.addHandler(c_handler)
logger.addHandler(f_handler)
#%%

# @pytest.mark.skip()
@log_request()
def get(payloads)-> Any:
    payloads = list(map( lambda x: json.dumps({'text': x}), payloads))
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

