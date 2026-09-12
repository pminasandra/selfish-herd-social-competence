# Pranav Minasandra
# pminasandra.github.io
# December 09, 2024

from pathlib import Path

#Directories
PROJECTROOT = Path(open(".cw", "r").read().rstrip())
DATA = PROJECTROOT / "Data"
FIGURES = PROJECTROOT / "Figures"

formats=['png', 'pdf', 'svg']

# Program flow
RUN_SIMS = False
CONDUCT_HUNGERGAMES = False
ANALYSE_DATA = True
ANALYSE_HUNGERGAMES = False


# Gradient descent config
GRAD_DESC_DX = 0.005
GRAD_DESC_DY = 0.005
GRAD_DESC_MAX_STEP_SIZE = 0.05
GRAD_DESC_MULTPL_FACTOR = 0.1

# Strucuring
MU = -1#momentum model

POP_S_DOR = {
 10: [0, 1, 2, 3, MU],
 25: [0, 1, 2, 3, MU],
 50: [0, 1, 2, 3, MU],
 35: [0, 1, 2, 3, MU],
 75: [0, 1, 2, 3, MU],
 87: [0, 1, 2, 3, MU],
100: [0, 1, 2, 3, MU]
} 
NUM_REPEATS = 500
TMAX = 500

# Program flow for hungergames
POP_S_SMART_GUYS_HG = {
    25: [5, 10, 15, 20],
    87: [5, 25, 45, 65]
} # these are how many d1 individuals to have in each round

HUNGERGAMES_TIME_LIMS = (250, 350)

# Data analysis
ANALYSE_POP_SIZES = [10, 25, 35, 50, 75, 87, 100]
ANALYSE_DEPTHS = [0, 1, 2, 3, MU]

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
