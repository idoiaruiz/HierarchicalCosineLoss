import logging
import os

import torch.distributed as dist

LOGGER_NAME = 'training_logger'


def setup(log_path, logs_filename='logs.log', debug=False):
    if debug:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(lvl)
    filename = os.path.join(log_path, logs_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(lvl)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(filename=filename)
    f_handler.setLevel(lvl)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)


def log_debug(msg, only_rank0=True):
    if only_rank0:
        rank = dist.get_rank()
        if rank != 0:
            return
    logger = logging.getLogger(LOGGER_NAME)
    logger.debug(msg)


def log_info(msg, only_rank0=True):
    if only_rank0:
        rank = dist.get_rank()
        if rank != 0:
            return
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(msg)


def log_error(msg, only_rank0=True):
    if only_rank0:
        rank = dist.get_rank()
        if rank != 0:
            return
    logger = logging.getLogger(LOGGER_NAME)
    logger.error(msg, exc_info=True)
