import matplotlib.pyplot as plt
import numpy as np
import statistics as stats

from utils import train_ensemble_method

if __name__ == "__main__":

	imblearn_datasets = [
		'ecoli',
		'spectrometer',
		'libras_move',
		'arrhythmia'
	]

	algorithms = ['DERSBoost']

	rend = np.empty((1,len(imblearn_datasets)))
	execution_times = []
	for i,algorithm in enumerate(algorithms):
		times = []
		for j,dataset in enumerate(imblearn_datasets):
			print(dataset.upper()+' with '+algorithm)
			r, IR, exec_time = train_ensemble_method(dataset,algorithm)
			rend[i,j] = r
			times.append(exec_time)
			print()
		execution_times.append(stats.mean(times))


	plt.figure(figsize=(10,10))
	plt.plot(imblearn_datasets, rend[0,:], label = 'DERS-Boost')
	plt.xticks(imblearn_datasets, size = 'small', rotation = 45)
	plt.legend(loc='upper right')
	plt.ylabel('Rendimiento')
	plt.xlabel('datasets')
	plt.show()

	print("Media del rendimiento para todos los datasets con DERSBoost: ",stats.mean(rend[0]))
