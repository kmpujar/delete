import json
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from feature import Features
import io
import codecs
from sklearn.model_selection import cross_val_score
import pandas
#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt12
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_fscore_support
import spacy
from sklearn.metrics import classification_report
nlp = spacy.load("en")
from nltk.corpus import stopwords
import nltk
def train_tfidf():

	return None

#need to define a proper structure for these functions

def train_baseline():
	with open('iswctotal.json', 'r') as f:  # load the sentences
		data = json.load(f)

	X = []
	for i in range(len(data)):
		X.append(data[i]['features'])
	temp = sklearn.preprocessing.normalize(X[:, [3]], axis=0)
	for i in range(len(temp)):
		X[i][3] = temp[i]

	#clf = RandomForestClassifier(random_state=0)

	clf = joblib.load('RF.pkl')

	prediction = clf.predict(X)

	for i in range(len(prediction)):
		data[i]['output'] = prediction[i]

	with open('predicted.json', 'w') as f:
		json.dump(data, f)

	return None #not sure about returning


#this function trains using fever
#need to do some cleaning

#return value not clear

'''
def training(config='baseline'):

	reads the dataset (training) file, extract features and train the fact-checking model
	:return: the model's performance


	data = np.array(fever())  # gets features from fever dataset

	X, x = sklearn.model_selection.train_test_split(data, test_size=0.3)  #test/train ratio

	Y = X[:, [X.shape[1] - 1]]
	y = x[:, [x.shape[1] - 1]]
	X = X[:, 0:X.shape[1] - 1]
	x = x[:, 0:x.shape[1] - 1]

	temp = sklearn.preprocessing.normalize(X[:, [3]], axis=0)
	for i in range(len(temp)):
		X[i][3] = temp[i]

	temp = sklearn.preprocessing.normalize(x[:, [3]], axis=0)
	for i in range(len(temp)):
		x[i][3] = temp[i]

	clf = RandomForestClassifier(random_state=0)
	clf.fit(X, Y)

	joblib.dump(clf, 'RF.pkl') #change name before running



	try:

		if config == 'baseline':
			train_baseline()
		elif config == 'tfidf':
			train_tfidf()
		else:
			# here we keep adding separated functions
			raise Exception('not supported: ' + config)

	except:
		raise
'''
def training_RF(dt,label, config='baseline'):
	'''
#X = sklearn.preprocessing.scale(data)
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=0)

	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]

	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	clf = RandomForestClassifier(n_estimators=25, random_state=40,oob_score=True)
	clf.fit(X_train, np.ravel(y_train))
	predict_labels=clf.predict(X_test)
	print(predict_labels)
	print("RF accuracy: "+ str(clf.score(X_test,np.ravel(y_test))))
	'''
	print(dt[0])
	data=np.array(dt)
	print(data[0])
	X, x , Y, y= sklearn.model_selection.train_test_split(data,label, test_size=0.3)  #test/train ratio

	#Y = X[:, [X.shape[1] - 1]]
	#y = x[:, [x.shape[1] - 1]]
	X = X[:, 0:X.shape[1] ]
	x = x[:, 0:x.shape[1] ]
	temp = sklearn.preprocessing.normalize(X[:, [3]], axis=0)
	for i in range(len(temp)):
		X[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(x[:, [3]], axis=0)
	for i in range(len(temp)):
		x[i][3] = temp[i]
	#print(x)
	#print(X[0])
	clf = RandomForestClassifier(random_state=0,n_estimators=100)
	clf.fit(X, np.ravel(Y))
	print("RF classsifier: ", clf.score(x, np.ravel(y)))
	scores = cross_val_score(clf, X, Y, cv=6)
	print("Cross-validated scores:", scores)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
						n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
		An object of that type which is cloned for each validation.

	title : string
		Title for the chart.

	X : array-like, shape (n_samples, n_features)
		Training vector, where n_samples is the number of samples and
		n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
		Target relative to X for classification or regression;
		None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
		Defines minimum and maximum yvalues plotted.

	cv : int, cross-validation generator or an iterable, optional
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:
		  - None, to use the default 3-fold cross-validation,
		  - integer, to specify the number of folds.
		  - An object to be used as a cross-validation generator.
		  - An iterable yielding train/test splits.

		For integer/None inputs, if ``y`` is binary or multiclass,
		:class:`StratifiedKFold` used. If the estimator is not a classifier
		or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

		Refer :ref:`User Guide <cross_validation>` for the various
		cross-validators that can be used here.

	n_jobs : int or None, optional (default=None)
		Number of jobs to run in parallel.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.

	train_sizes : array-like, shape (n_ticks,), dtype float or int
		Relative or absolute numbers of training examples that will be used to
		generate the learning curve. If the dtype is float, it is regarded as a
		fraction of the maximum size of the training set (that is determined
		by the selected validation method), i.e. it has to be within (0, 1].
		Otherwise it is interpreted as absolute sizes of the training sets.
		Note that for classification the number of samples usually have to
		be big enough to contain at least one sample from each class.
		(default: np.linspace(0.1, 1.0, 5))
	"""
	plt12.figure()
	plt12.title(title)
	if ylim is not None:
		plt12.ylim(*ylim)
	plt12.xlabel("Training examples")
	plt12.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt12.grid()

	#plt12.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color="r")
	#plt12.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt12.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt12.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt12.legend(loc="best")
	return plt12

def training_svm2(Data, label):
	Data=np.array(Data)
	label=np.array(label)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3, random_state=0)  #test/train ratio
	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]
	lsvm = LinearSVC()
	lsvm.fit(X_train,np.ravel(y_train))
	score=lsvm.score(X_test,np.ravel(y_test))
	print("SVM accuracy: "+str(score))

def training_svm(Data,label):
	#X=sklearn.preprocessing.normalize(Data, norm='l2')
	label=np.array(label)
	Data=np.array(Data)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3, random_state=100)  #test/train ratio
	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]

	clf = RandomForestClassifier(random_state=40, n_estimators=25)
	clf.fit(X_train, np.ravel(y_train))
	print("RF classsifier: ", clf.score(X_test, np.ravel(y_test)))
	op=clf.predict(X_test)
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	for i in range(0,len(op)):
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	#scores = cross_val_score(clf, X_test, y_test, cv=6)
	#print("Cross-validated scores:", scores)

	plot_learning_curve(clf, "Random Forest", X_train, y_train, cv=2, n_jobs=-1)

	# lsvm = LinearSVC()
	# lsvm.fit(X_train,np.ravel(y_train))
	# score=lsvm.score(X_test,np.ravel(y_test))
	# print("SVM accuracy: "+str(score))
	#
	# scores = cross_val_score(lsvm, X_train, y_train, cv=6)
	# print("Cross-validated scores:", scores)
	#
	# plot_learning_curve(lsvm, "Linear SVM", X_train, y_train, cv=5, n_jobs=-1)
	# op=clf.predict(X_test)
	# ct=0
	# ct0=0
	# count00=0
	# count01=0
	# count10=0
	# count11=0
	# for i in range(0,len(op)):
	# 	if y_test[i]==0:
	# 		ct0+=1
	# 		if(op[i]==0):
	# 			count00+=1
	# 		elif(op[i]==1):
	# 			count01+=1
	# 	elif y_test[i]==1:
	# 		ct+=1
	# 		if(op[i]==1):
	# 			count11+=1
	# 		elif(op[i]==0):
	# 			count10+=1
	# print("TN, FN, TP, FP", count00,count01,count11,count10)
	# print("neg score",float(count00)/ct0)
	# print("pos score",float(count11)/ct)

	mlp=MLPClassifier(hidden_layer_sizes=(110), activation='relu', batch_size='auto', learning_rate_init=0.001, max_iter=30000, validation_fraction=0.1, verbose=False, n_iter_no_change=200)
	mlp.fit(X_train, y_train)
	print("MLP accuracy "+str(mlp.score(X_test,np.ravel(y_test))))
	op=clf.predict(X_test)
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	for i in range(0,len(op)):
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	plot_learning_curve(mlp, "Neural Net", X_train, y_train, cv=2, n_jobs=-1)
	#parameters = {'kernel':('linear', 'rbf'), 'C':[0.10, 0.1, 10, 100, 1000], 'probability':[True, False], 'coef0':[0.0,0.05,0.1]}
	#svc = svm.SVC(gamma="scale")
	#clf = GridSearchCV(svc, parameters, cv=6, n_jobs=-1)
	#clf.fit(X_train,np.ravel(y_train))
	#print(clf.best_params_)
	#print(clf.best_score_ )
	#print(clf.best_estimator_)

	clf=svm.SVC(C=100, kernel='rbf',probability=True, gamma='scale', coef0=0.0)
	clf.fit(X_train,np.ravel(y_train))
	print("SVM_rbf"+str(clf.score(X_test,np.ravel(y_test))))
	op=clf.predict(X_test)
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	for i in range(0,len(op)):
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	plot_learning_curve(clf, "RBF SVM", X_train, y_train, cv=2, n_jobs=-1)
	plt.show()

def train_MLP(Data,label):
	print(Data[0])
	label=np.array(label)
	Data=np.array(Data)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3,random_state=100)  #tet/train ratio
	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]
		ik=11
	activ=['relu']
	while ik<110:
		ik+=11
		for act in activ:
			mlp=MLPClassifier(hidden_layer_sizes=(ik,ik), activation=act, batch_size='auto', learning_rate_init=0.001, max_iter=30000, validation_fraction=0.2, verbose=True, n_iter_no_change=250)
			mlp.fit(X_train, y_train)
			#print("MLP accuracy "+str(mlp.score(X_train,np.ravel(y_train))))
			print("MLP accuracy "+str(mlp.score(X_test,np.ravel(y_test)))+"	n:"+str(ik)+"a:"+act)
			#plot_learning_curve(mlp, "Neural Net", X_train, y_train, cv=5, n_jobs=-1)
			op=mlp.predict(X_test)
			#print(accuracy_score(y_train,mlp.predict(X_train)))
			ct=0
			ct0=0
			count00=0
			count01=0
			count10=0
			count11=0
			acc=0
			for i in range(0,len(op)):
				if op[i]==y_test[i]:
					acc+=1
				if y_test[i]==0:
					ct0+=1
					if(op[i]==0):
						count00+=1
					elif(op[i]==1):
						count01+=1
				elif y_test[i]==1:
					ct+=1
					if(op[i]==1):
						count11+=1
					elif(op[i]==0):
						count10+=1
			print("TN, FN, TP, FP", count00,count01,count11,count10)
			print("neg score",float(count00)/ct0)
			print("pos score",float(count11)/ct)
			print("accuracy",float(acc)/len(op))
			print(precision_recall_fscore_support(y_test, op, average=None))
	# plot_learning_curve(mlp, "Neural Net", X_train, y_train, cv=5, n_jobs=-1)
	# plt.show()

# 	clf=svm.SVC(C=100, kernel='rbf',probability=True, gamma='scale', coef0=0.0)
# 	clf.fit(X_train,np.ravel(y_train))#.score(X_train, np.ravel(y_train))
# 	print("SVM_rbf"+str(clf.score(X_test,np.ravel(y_test))))
# 	#plot_learning_curve(clf, "svm", X_train, y_train, cv=5, n_jobs=-1)
# 	#plt.show()
# 	op=clf.predict(X_test)
# 	#print(accuracy_score(y_train,mlp.predict(X_train)))
# 	ct=0
# 	ct0=0
# 	count00=0
# 	count01=0
# 	count10=0
# 	count11=0
# 	acc=0
# 	for i in range(0,len(op)):
# 		if op[i]==y_test[i]:
# 			acc+=1
# 		if y_test[i]==0:
# 			ct0+=1
# 			if(op[i]==0):
# 				count00+=1
# 			elif(op[i]==1):
# 				count01+=1
# 		elif y_test[i]==1:
# 			ct+=1
# 			if(op[i]==1):
# 				count11+=1
# 			elif(op[i]==0):
# 				count10+=1
# 	print("TN, FN, TP, FP", count00,count01,count11,count10)
# 	print("neg score",float(count00)/ct0)
# 	print("pos score",float(count11)/ct)
# 	print("accuracy",float(acc)/len(op))
# 	print(precision_recall_fscore_support(y_test, op, average=None))
#
# def training_RF1(X,y):
# 	#X=sklearn.preprocessing.normalize(X, norm='l1')
# 	data=np.array(X)
# 	y=np.array(y)
# 	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=100)
# 	X_train= X_train[:, 0:X_train.shape[1] ]
# 	X_test = X_test[:, 0:X_test.shape[1] ]
# 	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
# 	for i in range(len(temp)):
# 		X_train[i][3] = temp[i]
# 	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
# 	for i in range(len(temp)):
# 		X_test[i][3] = temp[i]
# 	print(X_test[0])
# 	print(X_train[0])
# 	clf = RandomForestClassifier(random_state=40, n_estimators=25)
# 	clf.fit(X_train, np.ravel(y_train))
# 	predict_labels=clf.predict(X_test)
# 	print("RF accuracy: "+ str(clf.score(X_test,y_test)))
# 	op=clf.predict(X_test)
# 	#print(accuracy_score(y_train,mlp.predict(X_train)))
# 	ct=0
# 	ct0=0
# 	count00=0
# 	count01=0
# 	count10=0
# 	count11=0
# 	acc=0
# 	for i in range(0,len(op)):
# 		if op[i]==y_test[i]:
# 			acc+=1
# 		if y_test[i]==0:
# 			ct0+=1
# 			if(op[i]==0):
# 				count00+=1
# 			elif(op[i]==1):
# 				count01+=1
# 		elif y_test[i]==1:
# 			ct+=1
# 			if(op[i]==1):
# 				count11+=1
# 			elif(op[i]==0):
# 				count10+=1
# 	print("TN, FN, TP, FP", count00,count01,count11,count10)
# 	print("neg score",float(count00)/ct0)
# 	print("pos score",float(count11)/ct)
# 	print("accuracy",float(acc)/len(op))
# 	print(precision_recall_fscore_support(y_test, op, average=None))
def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				  results['mean_test_score'][candidate],
				  results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			print("")

def grid_search(Data,label):
	label=np.array(label)
	Data=np.array(Data)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3,random_state=20)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]



	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
					 'C': [1, 10, 100]},
					{'kernel': ['linear'], 'C': [1, 10, 100]}]

	scores = ['precision', 'recall']

	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()


		clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,scoring='%s_macro' % score)
		clf.fit(X_train, y_train)

		print("Best parameters set found on development set for SVM:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean, std * 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		print()


	mlp = MLPClassifier(max_iter=250)
	parameter_space = {'hidden_layer_sizes': [(11,11), (44,44), (11,),(11,6)],'activation': ['tanh', 'relu'],'solver': ['sgd', 'adam'],'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive'],}
	clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
	clf.fit(X_train, y_train)
	# Best paramete set
	print('Best parameters found:\n', clf.best_params_)
		# All results
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	y_true, y_pred = y_test , clf.predict(X_test)


	print('Results on the test set:')
	print(classification_report(y_true, y_pred))

	clf = RandomForestClassifier()
		# use a full grid over all parameters
	param_grid = {"max_depth": [10,25,50],
				  "max_features": ['sqrt','log2'],
				  "min_samples_split": [2, 5, 15],
				  "criterion": ["gini", "entropy"],
				  "n_estimators":[10,20]}

		# run grid search
	grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
	grid_search.fit(X_train, y_train)

	print(len(grid_search.cv_results_['params']))
	report(grid_search.cv_results_)
	print('Best parameters found:\n', grid_search.best_params_)
		# All results
	means = grid_search.cv_results_['mean_test_score']
	stds = grid_search.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	y_true, y_pred = y_test , grid_search.predict(X_test)


	print('Results on the test set:')
	print(classification_report(y_true, y_pred))


def train_and_plot(Data,label):

	plt12.figure()
	plt12.title("FEVER 3-class")
	# if ylim is not None:
	# 	plt12.ylim(*ylim)
	plt12.xlabel("Training examples")
	plt12.ylabel("Score")

	print(Data[0])
	label=np.array(label)
	Data=np.array(Data)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3,random_state=100)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]

	#mlp=MLPClassifier(hidden_layer_sizes=(44,44), activation='relu', batch_size='auto',solver='adam', learning_rate='adaptive', max_iter=20000, validation_fraction=0.2, verbose=False, n_iter_no_change=300)
	mlp=MLPClassifier(hidden_layer_sizes=(44,44), activation='tanh',solver='adam', batch_size='auto', alpha=0.05, learning_rate='adaptive', max_iter=15000, validation_fraction=0.2, verbose=False, n_iter_no_change=200)
	mlp.fit(X_train, y_train)
	print("MLP accuracy "+str(mlp.score(X_test,np.ravel(y_test))))
	op=mlp.predict(X_test)
	print('Results on the test set:')
	print(classification_report(y_test, op))
	print("Neural Net")
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	acc=0
	for i in range(0,len(op)):
		if op[i]==y_test[i]:
			acc+=1
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	print("accuracy",float(acc)/len(op))
	print(precision_recall_fscore_support(y_test, op, average='micro'))
	#plot_learning_curve(mlp, "Neural Net", X_train, y_train, cv=5, n_jobs=-1)
	#plt.show()
	train_sizes, train_scores, test_scores = learning_curve(mlp, X_train, y_train, cv=5, n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5))
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt12.grid()

	plt12.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color="b")
	plt12.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="b")
	plt12.plot(train_sizes, train_scores_mean, '--', color="b",label="MLP Training score")
	plt12.plot(train_sizes, test_scores_mean, 'o-', color="b",label="MLP Cross-validation score")



	clf=svm.SVC(random_state=40, C=100, kernel='rbf',probability=True, gamma=0.001, coef0=0.0)
	clf.fit(X_train,np.ravel(y_train))#.score(X_train, np.ravel(y_train))
	print("SVM_rbf"+str(clf.score(X_test,np.ravel(y_test))))
	op=clf.predict(X_test)
	print('Results on the test set:')
	print(classification_report(y_test, op))
	print("SVM")
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	acc=0
	for i in range(0,len(op)):
		if op[i]==y_test[i]:
			acc+=1
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	print("accuracy",float(acc)/len(op))
	print(precision_recall_fscore_support(y_test, op, average='micro'))
	# plot_learning_curve(clf, "SVM", X_train, y_train, cv=5, n_jobs=-1)
	# plt.show()
	train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5))
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)


	plt12.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="m")
	plt12.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="m")
	plt12.plot(train_sizes, train_scores_mean, '--', color="m",label="SVM Training score")
	plt12.plot(train_sizes, test_scores_mean, 'o-', color="m",label="SVM Cross-validation score")




	rf = RandomForestClassifier(random_state=40, criterion= 'gini', max_depth= 10, max_features='log2', min_samples_split= 5, n_estimators= 20)
	rf.fit(X_train, np.ravel(y_train))
	print("RF accuracy: "+ str(clf.score(X_test,y_test)))
	op=rf.predict(X_test)
	print('Results on the test set:')
	print(classification_report(y_test, op))
	print("Random Forest")
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	acc=0
	for i in range(0,len(op)):
		if op[i]==y_test[i]:
			acc+=1
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	print("accuracy",float(acc)/len(op))
	print(precision_recall_fscore_support(y_test, op, average='micro'))
	# plot_learning_curve(rf, "Random Forest", X_train, y_train, cv=5, n_jobs=-1)
	# plt.show()
	train_sizes, train_scores, test_scores = learning_curve(mlp, X_train, y_train, cv=5, n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5))
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)


	plt12.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="c")
	plt12.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="c")
	plt12.plot(train_sizes, train_scores_mean, '--', color="c",label="RF Training score")
	plt12.plot(train_sizes, test_scores_mean, 'o-', color="c",label="RF Cross-validation score")

	plt12.legend(loc="best")
	#plt12.show()
	plt12.savefig('3class.png')

	# eclf = VotingClassifier(estimators=[('rf', mlp), ('svm', clf), ('mlp', rf)], voting='hard')
	# for clf, label in zip([mlp, clf, rf, eclf], ['Random Forest', 'SVM','Neural Net', 'Ensemble']):
	# 	scores = cross_val_score(clf,  X_train, y_train, cv=5, scoring='accuracy')
	# 	print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
	#
	# plot_learning_curve(eclf, "Ensemble", X_train, y_train, cv=5, n_jobs=-1)
	# plt12.show()


if __name__ == "__main__":
	train_data=[]
	label=[]
	json_data = open('fever_3.json')
	data = json.load(json_data)
	f=Features()
	#f.word2vecModel()
	c=0
	label1=[]
	label2=[]
	for d in data:
		if c<15:
			label1.append(d['label'])
			if len(d['triples'])>0:
				if(len(d['sentence'])>0 and len(d['triples'][0][0])>0 and len(d['triples'][0][1])>0 and len(d['triples'][0][2])>0):
					try:
						train_data.append(f.extract_features(d['sentence'], d['triples'][0][0], d['triples'][0][1],  d['triples'][0][2]))
					except:
						continue
					if d['label']==0: #or d['label']==0:
						label.append(0)
					elif d['label']==2:
						label.append(2)
					else:
						label.append(1)
					c=+1
	print(len(label))
	# print(len(label1))
	# print(len(label2))
	print(len(data))
	train_and_plot(train_data, label)
	#grid_search(train_data, label)
	'''
		if(d['label'])!=2 :
				if(d['spo'][2]!='' and c<5):
					#if(d['label']==0 and c<15000 or d['label']==1 ):
						#train_data.append(f.extract_features(d['body'], d['spo'][0],d['spo'][1], d['spo'][2]))
						label.append(d['label'])
						#print(train_data[c])
						#print(d)
						if(d['label']==1):
							print(d)
							print(train_data[c])
						wlist={}
						stop_words = set(stopwords.words('english'))
						shsh=d['body'].lower()#.replace('.','').replace(',','')

						shsh=' '.join(w for w in shsh.split() if w not in stop_words)
						sent_text = nltk.sent_tokenize(shsh)
						for f1 in sent_text:
							doc=nlp(f1)
							print(doc)
							for word in doc:
								wlist[str(word)]=str(list(word.children))
						print(wlist)
						break
						c+=1


	'''
	exit(0)
