class Config:
    """Set default configuration vars."""
	K_FOLDCV = 5

	RANDOM_STATE = None

    BASE_ESTIMATOR =  DecisionTreeClassifier(criterion='entropy', max_depth=1)
	N_ESTIMATORS = 10

    SMOTE_KNEIGHBORS = 5   
    
	MAJ_CLASS = -1
	MIN_CLASS = 1

#DERS-BOOST
	H = 6
	rus_selection_p = 0.4
	p = 0.2
	alpha = 0.8

#EVOLUTIONARY ALGORITHMS CONFIG
	CR = 0.6
	F = 0.5
	POP_SIZE = 10
	NGEN = 50
