import ast
import os
from configparser import ConfigParser

import torch
import torch.distributed as dist


_DEFAULT = './config_example.ini'


class CustomConfigParser(ConfigParser):
    def getlist(self, section, option, **kwargs):
        return ast.literal_eval(self.get(section, option, **kwargs))


def read_config_file(file_path):
    config = CustomConfigParser()
    config.read([_DEFAULT, file_path])

    return config


def all_reduce_dict(d, op=dist.ReduceOp.SUM):
    keys, values = [], []
    for k, v in d.items():
        keys.append(k)
        values.append(v)

    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op)
    values = torch.chunk(values, values.size(0), dim=0)
    new_d = {k: v.squeeze(0) for (k,v) in zip(keys, values)}
    return new_d


def mk_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

