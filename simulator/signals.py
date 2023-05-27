"""!
\brief Submodule dedicated to signals manipulation
"""

from numba import njit
import numpy             as np
import matplotlib.pyplot as plt

from . import util
from . import prob

@njit
def composeSignals(
    signal1, signal2,
    offset,
    amp1, amp2
):
    """!
    \brief Performs a composition of signals

    \param signal1 - first signal
    \param signal2 - second signal
    \param offset  - offset of the second signal relative to the first one
    \param amp1    - first signal multiplier
    \param amp2    - second signal multiplier

    \throws RuntimeError if signal grids aren't aligned
    """
    signal = ( signal1[0].copy(), signal1[1].copy() * amp1 )
    for E2 in signal2[0]:
        index2 = np.where(signal2[0] == E2)[0][0]
        E1 = E2 + offset
        if E1 in signal1[0]:
            index1 = np.where(signal[0] == E1)[0][0]
            signal[1][index1] += signal2[1][index2] * amp2
        elif E1 >= np.min(signal1[0]) and E1 <= np.max(signal1[0]):
            raise RuntimeError("Signal grids aren't aligned")
    
    return signal

@njit
def integrateSignal(signal, intFrom, intTo):
    """!
    \brief "Integrates" the given signal in the interval

    Integration is performed by summing signal's `Y` values

    \param signal  - given signal
    \param intFrom - _absolute_ left integration boundary
    \param intTo   - _absolute_ right integration boundary
    """
    return np.sum(
        signal[1][
            np.logical_and(
                signal[0] >= intFrom,
                signal[0] <= intTo
            )
        ]
    )

@njit
def integrateSignalRelative(signal, offsetLeft, offsetRight, center = 9):
    """!
    \brief "Integrates" the given signal in the interval
           given via offests relative to the point
           
    \param signal      - given signal
    \param offsetLeft  - left integration boundary offset relative to the center.
                         Same as intLeft = center - offsetLeft
    \param offsetRight - right integration boundary offset relative to the center.
                         Same as intRight = center + offsetRight
    \param center      - A point inside integration interval that offsets relate to
    """
    return integrateSignal(signal, center - offsetLeft, center + offsetRight)

@njit
def rollDoubleOverlap(P, E, signal, offsetLeft, offsetRight):
    """!
    \brief Perform a full roll for two-signal overlap

    Offset is determined as a uniformly distributed random integer from 0 to 42
    Amplitudes are determined as random values with distribution given by `P` and `E`

    \param P           - distribution P
    \param E           - distribution E
    \param signal      - signal shape
    \param offsetLeft  - left integration border offset relative to 9
    \param offsetRight - right integration border offset relative to 9

    See \ref integrateSignalRelative for better explaination of `offsetLeft` and `offsetRight`
    """
    offset = np.floor(np.random.uniform(0, 43))
    amp1 = prob.rollVal(P, E)
    amp2 = prob.rollVal(P, E)
    
    return np.array([
        offset,
        amp1,
        amp2,
        integrateSignalRelative(composeSignals(signal, signal, offset, amp1, amp2), offsetLeft, offsetRight)
    ])

class SignalTester:
    """!
    \brief Performs a bulk of simulation rolls on a signal and provides utility functions
    """
    def __init__(self, P, E, signal):
        """!
        \brief Initialze runner
        
        \param P, E   - amplitude distribution P, E
        \param signal - signal shape
        """
        self.P = P
        self.E = E
        self.signal = signal
        self.result = None
    
    def run(self, offsetLeft, offsetRight, numRolls=10_000_000, bulkNum=100):
        """!
        \brief Run `numRolls` double overlap simulations
        
        \param offsetLeft  - left border of integration offset relative to 9
        \param offsetRight - right border of integration offset relative to 9
        \param numRolls    - number of rolls
        \param bulkNum     - number of parallel processes
        """
        self.result = {
            "left":  offsetLeft,
            "right": offsetRight,
            "data":  np.array(
                util.runInParallelBulk(
                    bulkNum, numRolls / bulkNum, None,
                    rollDoubleOverlap,
                    self.P, self.E, self.signal,
                    offsetLeft, offsetRight
                )
            )
        }
    
    def runSingle(self, offsetLeft, offsetRight, numRolls=10_000_000, bulkNum=100):
        """!
        \brief Run `numRolls` single signal simulations
        
        \param offsetLeft  - left border of integration offset relative to 9
        \param offsetRight - right border of integration offset relative to 9
        \param numRolls    - number of rolls
        \param bulkNum     - number of parallel processes
        """
        amps = util.runInParallelBulk(
            bulkNum, numRolls / bulkNum, None,
            prob.rollVal, self.P, self.E
        )
        self.result = {
            "left":       offsetLeft,
            "right":      offsetRight,
            "dataSingle": np.array(
                [
                    integrateSignalRelative(
                        ( self.signal[0], self.signal[1] * amp ),
                        offsetLeft, offsetRight
                    )
                    for amp in amps
                ]
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
