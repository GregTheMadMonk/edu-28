"""!
\brief Submodule for various utility functions used in the project
"""

"""
\file util.py
\brief The file describes various utility functions used in the project
"""

import numpy             as np
import matplotlib.pyplot as plt
from numba           import njit
from multiprocessing import Pool

from . import prob

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

print("Initializing parallel tasks queue...")
tasks = [] # Parallel tasks queue

def parallelRunner(idx):
    """!
    \brief Runs a task from a task pool

    \param idx - index of the task to run
    """
    np.random.seed() # Each thread should have its own random seed
    return tasks[idx]()

def runInParallel(amount, poolSize, what, *args):
    """!
    \brief Runs in parallel several instances of the function

    This function is used to gather staticstics.
    It does not pass any index to the function.
    All of the runs are identical from the caller standpoint.

    \param amount   - final amount of runs
    \param poolSize - process pool size. Passed directly to `multiprocessing.Pool`.
                      Set to `None` to use automatic value.
    \param what     - a function (or functor) to run
    \param args     - function arguments

    kwargs aren't supported by numba
    """
    # Clean some of the old tasks
    global tasks
    while len(tasks) > 10:
        del tasks[0]
    
    # Add our task
    idx = len(tasks)
    tasks.append(lambda: what(*args))
    
    # Run threads
    with Pool(poolSize) as pool:
        return pool.map(
            parallelRunner,
            [ idx for i in range(amount) ]
        )

@njit
def bulkRunnerHelper(bulkSize, what, *args):
    """!
    \brief Returns a result of bulk-execution of function

    A helper for \ref runInParallelBulk
    """
    return [ what(*args) for i in range(bulkSize) ]

def runInParallelBulk(bulkNum, bulkSize, poolSize, what, *args):
    """!
    \brief Runs a bulk of cheap tasks in parallel

    'Cheap' tasks don't benefit from parallelization since the thread
    creationg cost exceeds the task cost.
    It is still possible to gain the advantage by concurrently running
    several sequential executions of such tasks.

    kwargs aren't supported by numba

    \param bulkNum  - number of tasks to process
    \param bulkSize - number of sequential runs of `what` in one task
    \param poolSize - see \ref runInParallel
    \param what     - see \ref runInParallel
    \param args     - see \ref runInParallel

    Total amount of runs equals `bulkNum * bulkSize`
    """
    return [
        elem for runResult in runInParallel(
            bulkNum, poolSize,
            bulkRunnerHelper, bulkSize, what, *args
        ) for elem in runResult
    ]

def loadExperimentalSignal(filename, separator='\t'):
    """!
    \brief Loads a signal shape from Numass experimental data file

    Uses tabs by default, changed with `separator`

    \param filename  - data file name
    \param separator - file column separator
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
    
    return ( np.array(signal[0]), prob.probNormalize(np.array(signal[0]), np.array(signal[1])) )

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
