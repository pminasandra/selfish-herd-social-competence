# Pranav Minasandra
# Dec 16, 2024
# pminasandra.github.io

"""
Had a slightly thoughtless way of saving data last time. Updated to be nicer.
Now fixing up old data. This file never needs to be run again.
"""

import glob
from os.path import join as joinpath
import os.path
import os
import pickle
import shutil

import numpy as np

import config

if __name__ == "__main__":
    depth_dirs = []
    for pop_size in config.POP_SIZES:
        depth_dirs.extend([joinpath(config.DATA, str(pop_size), f"d{depth}")\
                        for depth in config.DEPTHS_OF_REASONING])
    [os.makedirs(dir_, exist_ok=True) for dir_ in depth_dirs]

    prev_depth_dirs = [joinpath(config.DATA, f"d{depth}")\
                        for depth in config.DEPTHS_OF_REASONING]

    for ddir in prev_depth_dirs:
        files = glob.glob(joinpath(ddir, "*.pkl"))
        for file_ in files:
            with open(file_, "rb") as file_obj:
                data = pickle.load(file_obj)
                pop_size = data.shape[0]
                depth = os.path.basename(file_)[:-len(".pkl")].split("-")[-1]
                uname = "".join(os.path.basename(file_)\
                            [:-len(".pkl")].split("-")[:-1])

                dest = joinpath(config.DATA, str(pop_size), f"d{depth}", 
                                f"{pop_size}-{depth}-{uname}.pkl")

                print(file_, "->", dest)
                shutil.move(file_, dest)
