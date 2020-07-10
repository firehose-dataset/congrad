import numpy as np
import torch
import time
import functools
import os
import shutil

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600.0)
        if x >= 60:
            return '{:.1f}m'.format(x / 60.0)
        return '{}s'.format(round(x))

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    def reset(self):
        self.n = 0
        self.v = 0

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    run_id = None
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        run_id = 'resume'

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path,
                                  'resume.scripts' if run_id is not None else 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
            for src_file in scripts_to_save:
                dst_file = os.path.join(script_path, os.path.basename(src_file))
                print('copy {} to {}'.format(src_file, dst_file))
                if os.path.isdir(src_file):
                    shutil.copytree(src_file, dst_file)
                else:
                    shutil.copyfile(src_file, dst_file)

    log_path = os.path.join(dir_path,
                           'log.resume.txt'if run_id is not None else 'log.txt')
    return get_logger(log_path=log_path)
