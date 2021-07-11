import json
from utils import read_data, url, log_request, headers
import random
from typing import Any, Dict, List
from aiohttp import ClientSession
import asyncio


# Not save response data yet!
async def send(payload, session: ClientSession, **kwargs):
    resp = await session.request(method='GET', url=url, json=payload, **kwargs)
    # resp.raise_for_status()
    return resp

# TODO:: methods load the whole response in memory 
    # resp.read() for binary content 
    # resp.json() for json

# TODO: for large content 
#   resp.content.read(chunk_size)
@log_request(is_async=True)
async def send_multiple(payloads: List):
    async with ClientSession() as session:
        tasks = []
        for payload in payloads:
            tasks.append(
                    send(payload, session=session)
                    )
        
        await asyncio.gather(*tasks)
    pass

def test_async():
    payloads = read_data()
    random.shuffle(payloads)
    payloads = list(map( lambda x: {'text': x}, payloads))
    asyncio.run(send_multiple(payloads))
