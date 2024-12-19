# Pranav Minasandra and Cecilia Baldoni
# Dec 13, 2024
# pminasandra.github.io

import datetime as dt
from os.path import join as joinpath
import multiprocessing as mp
import os
import uuid

import numpy as np

import config
import selfishherd
import measurements

def runmodel(herd, filename):
    np.random.seed()
    herd.run(config.TMAX)
    herd.savedata(filename)

if __name__ == "__main__":
    if config.RUN_SIMS:
        depth_dirs = []
        for pop_size in config.POP_SIZES:
            depth_dirs.extend([joinpath(config.DATA, str(pop_size), f"d{depth}")\
                            for depth in config.DEPTHS_OF_REASONING])
        [os.makedirs(dir_, exist_ok=True) for dir_ in depth_dirs]

        for pop_size in config.POP_SIZES:
            print("Working on pop_size", pop_size)
            inits = [np.random.uniform(size=(pop_size, 2))\
                        for i in range(config.NUM_REPEATS)]
            init_names = [str(uuid.uuid4()) for i in range(config.NUM_REPEATS)]

            for depth in config.DEPTHS_OF_REASONING:
                print(dt.datetime.now(), "Depth of reasoning:", depth)
                herds = [selfishherd.SelfishHerd(pop_size, depth, loc) for loc\
                            in inits]
                filenames = [joinpath(config.DATA, str(pop_size), f"d{depth}",
                                    f"{pop_size}-{depth}-{uname}.pkl")\
                                    for uname in init_names]
                args = zip(herds, filenames)
                
                pool = mp.Pool(35)
                pool.starmap(runmodel, args)
                pool.close()
                pool.join()
            print()

    if config.ANALYSE_DATA:
        group_metrics = []
        timerange = range(0, 501, 20)
        os.makedirs(joinpath(config.DATA, "Results"), exist_ok=True)
        for pop_size in config.ANALYSE_POP_SIZES:
            for depth in config.ANALYSE_DEPTHS:
                tgs_file = joinpath(config.DATA, "Results",
                                        f"{pop_size}-d{depth}.csv")
                area_file = joinpath(config.DATA, "Results",
                                        f"areas-{pop_size}-d{depth}.csv")
                
                tgs_data = measurements.make_tgs_csv_for(pop_size, depth,
                                    timerange, eps=0.02)
                tgs_data.to_csv(tgs_file, index=False)
                
                area_data = measurements.make_area_csv_for(pop_size,
                                    depth, timerange)
                area_data.to_csv(area_file, index=False)

