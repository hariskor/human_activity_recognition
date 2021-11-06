import numpy as np
import matplotlib.pyplot as plt

import Preprocessor

if __name__ == '__main__':
    p = Preprocessor.Pipeline(loadPreprocessed=True, saveData=True, nFolds = 0)
    p.pipe()