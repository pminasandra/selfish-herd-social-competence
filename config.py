
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
RUN_SIMS = True
POP_SIZES = [50, 75, 100] #already have data for iterations 10, 25.
DEPTHS_OF_REASONING = [0, 1, 2, 3]
NUM_REPEATS = 500
TMAX = 500

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
