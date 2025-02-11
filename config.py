# Pranav Minasandra
# pminasandra.github.io
# December 09, 2024

import os
import os.path


#Directories
PROJECTROOT = open(".cw", "r").read().rstrip()
DATA = os.path.join(PROJECTROOT, "Data")
FIGURES = os.path.join(PROJECTROOT, "Figures")

formats=['png', 'pdf', 'svg']

# Gradient descent config
GRAD_DESC_DX = 0.005
GRAD_DESC_DY = 0.005
GRAD_DESC_MAX_STEP_SIZE = 0.05
GRAD_DESC_MULTPL_FACTOR = 0.1

# Program flow
RUN_SIMS = False
POP_S_DOR = {
10: [0, 1, 2, 3],
25: [0, 1, 2, 3],
50: [0, 1, 2, 3],
35: [0, 1, 2, 3],
75: [0, 1, 2, 3],
87: [0, 1, 2, 3],
100: [0, 1, 2, 3]
} 
NUM_REPEATS = 500
TMAX = 500

# Program flow for hungergames
CONDUCT_HUNGERGAMES = True
POP_S_SMART_GUYS_HG = {
    25: [5, 10, 15, 20],
    87: [5, 25, 45, 65]
} # these are how many d1 individuals to have in each round

# Data analysis
ANALYSE_DATA = False
ANALYSE_POP_SIZES = [10, 25, 35, 50, 75, 87, 100]
ANALYSE_DEPTHS = [0, 1, 2, 3]

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
