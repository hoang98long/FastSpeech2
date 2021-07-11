#%%
import requests
import json
from utils import read_data, url, log_request, headers
import random
from typing import Any, Dict
import logging

logging.getLogger('requests').disabled = True
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('sync_request.log')

logger.addHandler(c_handler)
logger.addHandler(f_handler)
#%%

# @pytest.mark.skip()
@log_request()
def send(payloads)-> Any:
    responses = [ requests.request('GET', url, headers=headers, data=p) for p in payloads ]
    # return response.content
    return responses

def test_request():
    # print('Inside test sync request')
    # import pdb; pdb.set_tra()
    payloads = read_data()
    random.shuffle(payloads)
    payloads = list(map( lambda x: json.dumps({'text': x}), payloads))
    # for _ in range(10):
    #     responses = send(payloads)
    responses = send(payloads)



    
if __name__ == '__main__':
    test_request()
    pass

