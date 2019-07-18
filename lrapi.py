from flask import Flask 
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

# pre-processing
dataset = pd.read_csv('https://raw.githubusercontent.com/kdhartmann/LinearModels/master/SaratogaHousesClean.csv')
y = dataset['price']
X = dataset.iloc[:, 1:9]

scaler = StandardScaler()
cv = KFold(n_splits=5, shuffle=False)
regIntercept = LinearRegression(fit_intercept = True)
reg = LinearRegression(fit_intercept = False)

# scaling the features 
XScaleTransform = scaler.fit_transform(X)
XScaled = pd.DataFrame(XScaleTransform, columns = X.columns)

# use in price = livingArea model for CV
livingArea = np.array(XScaled['livingArea'])
livingArea = livingArea.reshape(-1,1)

# K-fold: get x/y train/test points and their MSE
mseList = []
splitList = []
XTrainList = []
yTrainList = []
XTestList = []
yTestList = []
split = 1

# loop through the splits 
for train_index, test_index in cv.split(livingArea):
	X_train, X_test, y_train, y_test = livingArea[train_index], livingArea[test_index], y[train_index], y[test_index]
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	MSE = metrics.mean_squared_error(y_test, y_pred)
	mseList.append(MSE)
	# format the X train and test correctly 
	X_trainCorrect = []
	for elem in X_train:
		X_trainCorrect.append(elem[0])
	X_testCorrect = []
	for elem in X_test:
		X_testCorrect.append(elem[0])
	# append results to list to later be added to dataframe
	XTrainList.append(X_trainCorrect)
	yTrainList.append(np.array(y_train))
	XTestList.append(X_testCorrect)
	yTestList.append(np.array(y_test))
	splitList.append(split)
	split +=1

# create the dataframe for the user input graph results 
inputGraphResults = pd.DataFrame(columns = ['model', 'numFeat', 'mse', 'selectedFeat'])


# function to get dataframe of regression results: coefficients and features
def getResultsDF(model, features):
    df = pd.DataFrame()
    names = []
    coefs = []
    for elem in features:
        names.append(elem)
    for elem in model.coef_:
        coefs.append(elem)
    df['feature'] = names
    df['coef'] = coefs
    return df.reindex(df.coef.abs().sort_values(ascending = False).index)

def getResultsDFIntercept(model, features):
    df = pd.DataFrame()
    names = ['intercept']
    coefs = [round(model.intercept_,2)]
    for elem in features:
        names.append(elem)
    for elem in model.coef_:
        coefs.append(round(elem,2))
    df['feature'] = names
    df['coef'] = coefs
    return df

def findNextFeat(featIncluded, featExcluded):
    mseList = []
    global XScaled
    for elem in featExcluded:
        feats1 = featIncluded.tolist()
        feats1.append(elem)
        XFeats = XScaled[feats1]
        MSE = np.mean((cross_val_score(reg, XFeats, y, cv=cv, scoring='neg_mean_squared_error'))*-1)
        mseList.append(MSE)
    lowestMSE = min(mseList)
    lowestIndex = mseList.index(lowestMSE)
    lowestFeat = featExcluded[lowestIndex]
    
    return lowestFeat, lowestMSE


app = Flask(__name__)

## APIs

@app.route('/linearResultsUnscaled/<features>')
def linearResultsUnscaled(features):
	features = features.split(",")
	XlinUnscaled = X[features]
	regIntercept.fit(XlinUnscaled, y)
	results = getResultsDFIntercept(regIntercept, np.array(XlinUnscaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/linearResultsScaled/<features>')
def linearResultsScaled(features):
	features = features.split(",")
	XlinScaled = X[features]
	reg.fit(XlinScaled, y)
	results = getResultsDF(reg, np.array(XlinScaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/roomsUnscaled')
def roomsUnscaled():
	rooms = X[['rooms']]
	rooms_json = rooms.to_json(orient='records')
	return rooms_json

@app.route('/roomsScaled')
def roomsScaled():
	roomsScaled = XScaled[['rooms']]
	roomsScaled_json = roomsScaled.to_json(orient='records')
	return roomsScaled_json

# MSE from single train-test split
@app.route('/trainTestSplitMSE')
def trainTestSplitMSE():
	livingArea = XScaled[['livingArea']]
	X_train, X_test, y_train, y_test = train_test_split(livingArea, y, test_size=.2, random_state=1)
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	trainTestMSE = metrics.mean_squared_error(y_test, y_pred)
	trainTestMSE_dict = {"trainTestMSE": trainTestMSE}
	trainTestMSE_json = json.dumps(trainTestMSE_dict)
	return trainTestMSE_json

# dataframe of MSEs for the folds 
@app.route('/kfoldmse')
def kfoldmse():
	results = pd.DataFrame()
	results['fold'] = splitList
	results['mse'] = mseList
	results_json = results.to_json(orient='records')
	return results_json

# gives the train data for a specified fold
@app.route('/kfoldTrain/<fold>')
def kfoldTrain(fold):
	fold = int(fold)
	trainDF = pd.DataFrame()
	trainDF['XTrain'] = XTrainList[fold-1]
	trainDF['yTrain'] = yTrainList[fold-1]
	train_json = trainDF.to_json(orient='records')
	return train_json

# gives test data for a specified fold
@app.route('/kfoldTest/<fold>')
def kfoldTest(fold):
	fold = int(fold)
	testDF = pd.DataFrame()
	testDF['XTest'] = XTestList[fold-1]
	testDF['yTest'] = yTestList[fold-1]
	test_json = testDF.to_json(orient='records')
	return test_json

# takes in selectedFeat and concatenates results to inputGraphResults dataframe 
@app.route('/inputGraphs/<selectedFeats>')
def inputGraphs(selectedFeats):
	selectedFeats = selectedFeats.split(",")
	numFeat = len(selectedFeats)
	selectedFeatDF = pd.DataFrame()
	# create correctly structured string of features selected
	outputString = ''
	for elem in selectedFeats:
		selectedFeatDF[elem] = XScaled[elem]
		if len(outputString) == 0:
			outputString += elem
		else:
			outputString += (f", {elem}")
	MSE = (cross_val_score(reg, selectedFeatDF, y, cv=cv, scoring='neg_mean_squared_error'))*-1
	# create df to concatenate to inputGraphResults
	toConcatDF = pd.DataFrame()
	toConcatDF['numFeat'] = [numFeat]
	toConcatDF['mse'] = [np.mean(MSE)]
	toConcatDF['selectedFeat'] = [outputString]
	global inputGraphResults
	toConcatDF['model'] = (inputGraphResults['selectedFeat'].shape[0] + 1)
	inputGraphResults = pd.concat([inputGraphResults, toConcatDF], sort = True)
	inputGraphResults_json = inputGraphResults.to_json(orient='records')
	return inputGraphResults_json

# lowest MSE for each possible number of features
@app.route('/lowestMSEByFeatureCount')
def lowestMSEByFeatureCount():
	results = pd.DataFrame()
	featureNum = 1
	feats = []
	featuresList = []
	mseList = []
	numFeatList = []
	potentialFeats = XScaled.columns
	while featureNum <= 8:
		feats = np.asarray(feats)
		featsExcluded = np.setdiff1d(potentialFeats, feats)
		featsExported, lowestMSE = findNextFeat(feats, featsExcluded)  
		feats = np.append(feats,featsExported)
		outputString = ''
		for elem in feats:
			if len(outputString) == 0:
				outputString += elem
			else:
				outputString += (f", {elem}")
		featuresList.append(outputString)
		mseList.append(lowestMSE)
		numFeatList.append(feats.shape[0])
		featureNum += 1
    
	results['numFeat'] = numFeatList
	results['mse'] = mseList
	results['features'] = featuresList
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/linearResults')
def linearResults():
	reg.fit(XScaled, y)
	results = getResultsDF(reg, np.array(XScaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/featureSelectionResults/<numFeat>')
def featureSelectionResults(numFeat):
	numFeat = int(numFeat)
	KBest = SelectKBest(f_regression, k=numFeat)
	Kfit = KBest.fit_transform(XScaled, y)
	column_names = np.array(XScaled.columns[KBest.get_support()])
	reg.fit(Kfit,y)
	results = getResultsDF(reg, column_names)
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/lassoResults/<_lambda>')
def lassoResults(_lambda):
	_lambda = float(_lambda)
	lasso = Lasso(alpha=_lambda)
	lasso.fit(XScaled, y)
	results = getResultsDF(lasso, np.array(XScaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/ridgeResults/<_lambda>')
def ridgeResults(_lambda):
	_lambda = float(_lambda)
	ridge = Ridge(alpha=_lambda)
	ridge.fit(XScaled, y)
	results = getResultsDF(ridge, np.array(XScaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

if __name__ == '__main__':
    app.run(debug=False)