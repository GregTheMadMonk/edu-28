"""!
\brief Submodule for various utility functions used in the project
"""

"""
\file util.py
\brief The file describes various utility functions used in the project
"""

import numpy             as np
import matplotlib.pyplot as plt

from . import cpp

def plotSignals(*signals, **namedSignals):
    """!
    \brief Draws an arbitrary amount of signals on the same plot

    All arguments are treated as signals in format described above.
    Keyword arguments use their keywords as plot titles
    """
    for idx, signal in enumerate(signals):
        plt.plot(signal[0], signal[1], label=f"Signal {idx}")
    for name, signal in namedSignals.items():
        plt.plot(signal[0], signal[1], label=name)
    plt.legend()

def loadExperimentalSignal(filename, separator='\t', trimLength = 20):
    """!
    \brief Loads a signal shape from Numass experimental data file

    Uses tabs by default, changed with `separator`

    \param filename   - data file name
    \param separator  - file column separator
    \param trimLength - trim the signal after this value. Set to None to disable
    """
    signal = ( [], [] )
    with open(filename) as sigFile:
        header = True
        for line in sigFile.readlines():
            if header:
                header = False
                continue
            
            point = list(
                map(
                    lambda x: float(x) if x else 0,
                    line.split(separator)
                )
            )
            
            signal[0].append(point[0])
            signal[1].append(sum(point[1:]))
    
    E = np.array(signal[0])
    P = np.array(signal[1])

    if trimLength is not None:
        P = P[E <= trimLength]
        E = E[E <= trimLength]

    return ( E, cpp.get().probNormalize(E, P) )

def readHistFile(filename, separator=' '):
    """!
    \brief Reads a histogram file and returns it as a numpy array

    \param filename  - histogram file name
    \param separator - column separator
    """
    with open(filename) as histFile:
        lines = histFile.readlines()
        ret = np.zeros((len(lines), 2))
        
        for idx, line in enumerate(lines):
            if line == "":
                continue
            
            point = line.split(separator)
            
            for i in range(2):
                ret[idx][i] = float(point[i])
        
        return ret

def analyzeHistFile(filename, border, separator=' '):
    """!
    \brief Reads a histogram file and determines how many matches hit left or right of the border

    \param filename  - histogram file name
    \param border    - border
    \param separator - column separator
    """
    with open(filename) as histFile:
        left = 0
        right = 0
        
        for line in histFile.readlines():
            if line == "":
                continue
            
            point = line.split(separator)
            
            if float(point[0]) < border:
                left += float(point[1])
            else:
                right += float(point[1])
        
        return (left, right)
