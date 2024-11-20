from logging import getLogger
from recbole.utils import init_logger


def init_logger(config):

    init_logger(config)
    return getLogger()
