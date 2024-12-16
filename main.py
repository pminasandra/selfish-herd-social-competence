# Pranav Minasandra and Cecilia Baldoni
# Dec 13, 2024
# pminasandra.github.io


from os.path import join as joinpath
import multiprocessing as mp
import os
import uuid

import numpy as np

import config
import selfishherd

def runmodel(herd, filename):
    np.random.seed()
    herd.run(config.TMAX)
    herd.savedata(filename)

if __name__ == "__main__":
    if config.RUN_SIMS:
        depth_dirs = []
        for pop_size in config.POP_SIZES:
            depth_dirs.append([joinpath(config.DATA, pop_size, f"d{depth}")\
                            for depth in config.DEPTHS_OF_REASONING])
        [os.makedirs(dir_, exist_ok=True) for dir_ in depth_dirs]

        for p_size in config.POP_SIZES:
            pop_size = str(p_size)
            print("Working on pop_size", pop_size)
            inits = [np.random.uniform(size=(pop_size, 2))\
                        for i in range(config.NUM_REPEATS)]
            init_names = [str(uuid.uuid4()) for i in range(config.NUM_REPEATS)]

            for depth in config.DEPTHS_OF_REASONING:
                print("Depth of reasoning:", depth)
                herds = [selfishherd.SelfishHerd(pop_size, depth, loc) for loc\
                            in inits]
                filenames = [joinpath(config.DATA, pop_size, f"d{depth}",
                                    f"{pop_size}-{depth}-uname.pkl")\
                                    for uname in init_names]
                args = zip(herds, filenames)
                
                pool = mp.Pool(35)
                pool.starmap(runmodel, args)
                pool.close()
                pool.join()
            print()
