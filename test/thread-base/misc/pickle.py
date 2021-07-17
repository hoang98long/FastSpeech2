import zlib
import pickle


def compress(obj):
    p = pickle.dumps(obj)
    return zlib.compress(p)


def decompress(pickled):
    p = zlib.decompress(pickled)
    return pickle.loads(p)
