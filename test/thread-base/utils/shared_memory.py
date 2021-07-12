import os
import torch.multiprocessing as mp

MANAGER = mp.Manager()

def function_wrapper(f):

  def predict(*args, shared_dict=None):  # pylint: disable=syntax-error
    try:
      prediction = f(*args)
      if shared_dict is not None:
        shared_dict.update({"result": prediction})
    except Exception as e:
      shared_dict.update({"error": str(e)})

  return predict


def run_function(f, *args):
  try:
    shared_dict = MANAGER.dict()
    p = mp.Process(target=f, args=args, kwargs={"shared_dict": shared_dict}, daemon=True)
    p.start()
    p.join()
    res = dict(shared_dict)
    return res
  except Exception as e:
    return {"error": str(e)}
