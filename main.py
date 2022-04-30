import numpy as np
import matplotlib.pyplot as plt

# import Pipeline
import Ensemble

if __name__ == '__main__':
    p = Ensemble.Pipe(loadPreprocessed=False, trimToLength = 100)
    p.pipe()