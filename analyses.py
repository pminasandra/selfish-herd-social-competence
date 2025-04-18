# Pranav Minasandra and Cecilia Baldoni
# 27 Jan, 2025
# pminsandra.github.io

import os.path
from os.path import join as joinpath

from scipy.stats import linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import measurements
import utilities

RESULTS_DIR = joinpath(config.DATA, "Results")

trange = np.array(range(0, 481, 20))#(0, 481, 20) for speed etc
tcolnames = [f"t{i}" for i in trange]

keep_only = trange>100

def load_gsdata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"{popsize}-d{dval}.csv"))

def load_areadata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"areas-{popsize}-d{dval}.csv"))
    
def load_areavardata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"var-area-{popsize}-d{dval}.csv"))

def load_speeddata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"speed-{popsize}-d{dval}.csv"))

def load_eedata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"ee-{popsize}-d{dval}.csv"))

def get_avg_val(df):
    dfs = df[tcolnames]

    return dfs.mean(axis=0).to_numpy()

def get_cis_for(df, ulim=0.975, llim=0.025):
    dfs = df[tcolnames]

    dfs = dfs.to_numpy()
    res = np.quantile(dfs, (ulim, llim),
                method='closest_observation',
                axis=0)

    return res

if __name__ == "__main__":

    for popsize in config.ANALYSE_POP_SIZES:
        fig, ax = plt.subplots()
        for dval in config.ANALYSE_DEPTHS:
            gs = load_speeddata_for(popsize, dval)
            avg = get_avg_val(gs)
            lims = get_cis_for(gs)

            avg = avg
            lims=lims
            ulims = lims[0,:]
            llims = lims[1,:]

            trange_x = trange
            ax.plot(trange_x, avg, label=f"d{dval}")
            ax.fill_between(trange_x, llims, ulims, alpha=0.1)

        ax.legend()
        ax.set_ylabel("Mean speed of individuals")

        utilities.saveimg(fig, f"speed-{popsize}")
