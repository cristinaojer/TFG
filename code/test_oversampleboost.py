import matplotlib.pyplot as plt
import numpy as np
import statistics as stats

from utils import train_ensemble_method

if __name__ == "__main__":

	imblearn_datasets = [
		'ecoli',
		'optical_digits',
		'satimage',
		'pen_digits',
		'abalone',
		'sick_euthyroid',
		'spectrometer',
		'car_eval_34',
		'isolet',
		'us_crime',
		'yeast_ml8',
		'scene',
		'libras_move',
		'thyroid_sick',
		'coil_2000',
		'arrhythmia',
		'solar_flare_m0',
		'oil',
		'car_eval_4',
		'wine_quality',
		'letter_img',
		'yeast_me2',
		#     'webpage',
		'ozone_level',
		'mammography',
		#     'protein_homo',
		'abalone_19'
	]

	algorithms = ['SMOTEBoost','SMOTETomekBoost','SMOTEENNBoost']

	rend = np.empty((3,len(imblearn_datasets)))
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
	plt.plot(imblearn_datasets, rend[0,:], label = 'SMOTE-Boost')
	plt.plot(imblearn_datasets, rend[1,:], label = 'SMOTETomek-Boost')
	plt.plot(imblearn_datasets, rend[2,:], label = 'SMOTEENN-Boost')
	plt.xticks(imblearn_datasets, size = 'small', rotation = 45)
	plt.legend(loc='upper right')
	plt.ylabel('Rendimiento')
	plt.xlabel('datasets')
	plt.show()


	print("Media del rendimiento para todos los datasets con SMOTEBoost: ",stats.mean(rend[0]))
	print("Media del rendimiento para todos los datasets con SMOTETOMEK: ",stats.mean(rend[1]))
	print("Media del rendimiento para todos los datasets con SMOTEENN: ",stats.mean(rend[2]))