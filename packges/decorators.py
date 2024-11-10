from time import time
import logging

def timer(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.info(f"{func.__name__} took {end-start} seconds to run.")
        return result
    return wrapper