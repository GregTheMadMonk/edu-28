"""!
\brief Submodule with probability utilities
"""

import numpy as np
from numba import njit

def probNormalize(E, P):
    """!
    \brief Normalize probability distribution

    Makes it so that \int PdE = 1

    Returns new P
    """
    # Integrate P
    PInt = 0
    for i in range(len(E) - 1):
        PInt += (P[i] + P[i+1]) * (E[i+1] - E[i]) / 2
    return P/PInt

@njit
def rollVal(P, E):
    """!
    \brief Rolls a random value with a given probability distribution

    Assumes distribution defined by `P`, `E` is normalized
    """
    # Integrate probability density to get probability
    Pint = 0
    PintArr = np.full_like(P, 0)
    # assert(P[0] == 0)
    for i in range(1, len(P)): # Len preserves interoperability with Python arrays
        PintArr[i] = PintArr[i - 1] + P[i] * (E[i] - E[i - 1])
    # assert(np.isclose(PintArr[-1], 1))
    
    roll = np.random.uniform(0, 1, 1)[0] # Random roll
    
    # Convert our uniform roll into a random number, interpolate
    lowerIdx = np.where(PintArr <= roll)[0][-1]
    upperIdx = np.where(PintArr >= roll)[0][0]
    # assert(lowerIdx + 1 == upperIdx)
    
    t = (roll - PintArr[lowerIdx]) / (PintArr[upperIdx] - PintArr[lowerIdx])
    return E[lowerIdx] + t * (E[upperIdx] - E[lowerIdx])
