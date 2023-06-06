"""!
\brief Submodule dedicated to signals manipulation
"""

from numba import njit
import numpy             as np
import matplotlib.pyplot as plt

from . import cpp
from . import util

class SignalTester:
    """!
    \brief Performs a bulk of simulation rolls on a signal and provides utility functions
    """
    def __init__(self, P, E, signal):
        """!
        \brief Initialze runner
        
        \param P, E      - amplitude distribution P, E
        \param signal    - signal shape
        \param useCppMod - custom roll function
        """
        self.P = P
        self.E = E
        self.signal = signal
        self.result = None
    
    def run(self, offsetLeft, offsetRight, numRolls=10_000_000):
        """!
        \brief Run `numRolls` double overlap simulations
        
        \param offsetLeft  - left border of integration offset relative to 9
        \param offsetRight - right border of integration offset relative to 9
        \param numRolls    - number of rolls
        """
        self.result = {
            "left":  offsetLeft,
            "right": offsetRight,
            "data":  np.array(
                cpp.get().toList(
                    cpp.get().rollDoubleOverlapBulk(
                        numRolls,
                        self.E, self.P, self.signal,
                        offsetLeft, offsetRight
                    )
                )
            )
        }
    
    def runSingle(self, offsetLeft, offsetRight, numRolls=10_000_000):
        """!
        \brief Run `numRolls` single signal simulations
        
        \param offsetLeft  - left border of integration offset relative to 9
        \param offsetRight - right border of integration offset relative to 9
        \param numRolls    - number of rolls
        """
        self.result = {
            "left":       offsetLeft,
            "right":      offsetRight,
            "dataSingle": np.array(
                cpp.get().rollSingleBulk(numRolls, self.E, self.P, self.signal, offsetLeft, offsetRight)
            )
        }
    
    def plot(self, bins=1001, figsize=(10, 10), dump=None, dumpSep=' ', log=False, draw=True):
        """!
        \brief Creates a plot with the computaion result
        
        \param bins    - number of bins to use
        \param figsize - figsize to use
        \param dump    - file name to dump output histogram to
        \param dumpSep - dump separator
        \param log     - use logarithmic scale
        \param draw    - if `False` don't draw histogram, uses `numpy`'s hist instead of `matplotlib`'s hist

        \throws RuntimeError if `draw=True` and `dump` is not specified.
        """
        assert(self.result is not None)
        if (not draw) and (dump is None):
            raise RuntimeError("`draw=False` is meaningless with `dump=None`")
        
        hist = None
        
        if draw:
            if "data" in self.result:
                fig, ax = plt.subplots(2, 2, squeeze=True, figsize=figsize)
                for i in range(2):
                    for j in range(2):
                        ax[i, j].set_box_aspect(1)
                        if log: ax[i, j].set_yscale("log")

                titleBase = f"Channels {9-self.result['left']}-{9+self.result['right']}"

                hist = ax[0,0].hist(self.result["data"][:,3], bins=bins, density=True)
                ax[0,0].set_title(f"{titleBase}: integral distribution")

                ax[0,1].hist(
                    self.result["data"][:,0],
                    bins=int(
                        np.max(self.result["data"][:,0])
                        -
                        np.min(self.result["data"][:,0])
                    ) + 1
                )
                ax[0,1].set_title(f"{titleBase} - second peak offset distribution")

                ax[1,0].hist(self.result["data"][:,1], bins=bins, density=True)
                ax[1,0].set_title(f"{titleBase} - first peak amplitude distribution")
                ax[1,1].hist(self.result["data"][:,2], bins=bins, density=True)
                ax[1,1].set_title(f"{titleBase} - second peak amplitude distribution")
            elif "dataSingle" in self.result:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                if log: ax.set_yscale("log")
                hist = ax.hist(
                    self.result["dataSingle"],
                    bins=bins,
                    label=f"Integrating single signal channels {9-self.result['left']} through {9+self.result['right']}"
                )
                ax.legend()
        else:
            if "data" in self.result:
                hist = np.histogram(self.result["data"][:,3], bins=bins, density=True)
                
            elif "dataSingle" in self.result:
                hist = np.histogram(self.result["dataSingle"], bins=bins)
        
        if hist is not None and dump is not None:
            with open(dump, "w+") as histOutput:
                for n, x in zip(hist[0], hist[1]):
                    histOutput.write(f"{x}{dumpSep}{n}\n")
