import matplotlib.pyplot as plt
import numpy as np
import statistics as stats

from utils import train_ensemble_method

if __name__ == "__main__":
	train_ensemble_method('ecoli','DERSBoost')
