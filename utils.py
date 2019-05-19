import os
import sys
import logging


def make_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def custom_logger(name):
    # Taken from https://stackoverflow.com/questions/28330317/print-timestamp-for-logging-in-python
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)

    return logger