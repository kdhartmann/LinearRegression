from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression


app = Flask(__name__)
CORS(app)

# function to get dataframe of regression results: coefficients and features
# no intercept with sorting 
def getResultsDF(model, features):
    df = pd.DataFrame()
    names = ['intercept']
    coefs = [round(model.intercept_,2)]
    for elem in features:
        names.append(elem)
    for elem in model.coef_:
        coefs.append(round(elem,2))
    df['feature'] = names
    df['coef'] = coefs
    return df.reindex(df.coef.abs().sort_values(ascending = False).index)

# function to get dataframe of regression results: coefficients and features
# intercept without sorting 
def getResultsDFNoSort(model, features):
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

# finds the next best feature in featExcluded to add to featIncluded
def findNextFeat(featIncluded, featExcluded):
    mseList = []
    global XScaled
    for elem in featExcluded:
        featsList = featIncluded.tolist()
        featsList.append(elem)
        XFeats = XScaled[featsList]
        MSE = np.mean((cross_val_score(reg, XFeats, y, cv=cv, scoring='neg_mean_squared_error'))*-1)
        mseList.append(MSE)
    lowestMSE = min(mseList)
    lowestIndex = mseList.index(lowestMSE)
    lowestFeat = featExcluded[lowestIndex]
    
    return lowestFeat, lowestMSE

# pre-processing
dataset = pd.read_csv('SaratogaHousesClean.csv')
y = dataset['price']
X = dataset.iloc[:, 1:9]

scaler = StandardScaler()
cv = KFold(n_splits=5, shuffle=False)
reg = LinearRegression(fit_intercept = True)

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

# Finds Lowest MSE for each Feature Number
featureNum = 1
featsIncluded = []
featuresList = []
mseListLowest = []
numFeatList = []
potentialFeats = XScaled.columns
# loops through each possible number of features
while featureNum <= 8:
	# features used
	featsIncluded = np.asarray(featsIncluded)
	# finds features that aren't used yet 
	featsExcluded = np.setdiff1d(potentialFeats, featsIncluded)
	# function returns next feature and the MSE
	featsExported, lowestMSE = findNextFeat(featsIncluded, featsExcluded) 
	# add the next feature to the array of used features 
	featsIncluded = np.append(featsIncluded,featsExported)
	# format list into string of features 
	featsIncludedString = ''
	for elem in featsIncluded:
		if len(featsIncludedString) == 0:
			featsIncludedString += elem
		else:
			featsIncludedString += (f", {elem}")
	# append results to lists 
	featuresList.append(featsIncludedString)
	mseListLowest.append(lowestMSE)
	numFeatList.append(featsIncluded.shape[0])
	featureNum += 1

# create the dataframe for the user input graph results 
inputGraphResults = pd.DataFrame(columns = ['model', 'numFeat', 'mse', 'selectedFeat'])


## APIs

# regression results for unscaled features
# same input as linearResultsScaled
@app.route('/linearResultsUnscaled/<features>')
def linearResultsUnscaled(features):
	# creates a list of features
	features = features.split(",")
	# get data, fit regression with intercept, get df of results
	XlinUnscaled = X[features]
	reg.fit(XlinUnscaled, y)
	results = getResultsDFNoSort(reg, np.array(XlinUnscaled.columns))
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

# regression results for scaled features
# same input as linearResultsUnscaled
@app.route('/linearResultsScaled/<features>')
def linearResultsScaled(features):
	# creates list of features
	features = features.split(",")
	# get data, fit regression, get df of results
	XlinScaled = X[features]
	reg.fit(XlinScaled, y)
	results = getResultsDFNoSort(reg, np.array(XlinScaled.columns))
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

@app.route('/rooms', methods=['GET'])
def rooms():

	if request.args['scale']=='scaled':
		df = XScaled
	elif request.args['scale']=='unscaled':
		df = X

	return jsonify(
		df[['rooms']].to_dict(orient='records')
	)

# returns MSE from single train-test split
@app.route('/trainTestSplitMSE')
def trainTestSplitMSE():
	livingArea = XScaled[['livingArea']]
	X_train, X_test, y_train, y_test = train_test_split(livingArea, y, test_size=.2, random_state=1)
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	trainTestMSE = metrics.mean_squared_error(y_test, y_pred)
	trainTestMSE_dict = {"trainTestMSE": trainTestMSE}
	trainTestMSE_json = json.dumps(trainTestMSE_dict)
	return jsonify(trainTestMSE_json)

# returns dataframe of MSEs for all the folds that were calculated for mseList
@app.route('/kfoldmse')
def kfoldmse():
	results = pd.DataFrame()
	results['fold'] = splitList
	results['mse'] = mseList
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

# gives the train data for a specified fold from (X/y)trainList
@app.route('/kfoldTrain/<fold>')
def kfoldTrain(fold):
	fold = int(fold)
	trainDF = pd.DataFrame()
	trainDF['XTrain'] = XTrainList[fold-1]
	trainDF['yTrain'] = yTrainList[fold-1]
	train_json = trainDF.to_dict(orient='records')
	return jsonify(train_json)

# gives test data for a specified fold from (X/y)testList
@app.route('/kfoldTest/<fold>')
def kfoldTest(fold):
	fold = int(fold)
	testDF = pd.DataFrame()
	testDF['XTest'] = XTestList[fold-1]
	testDF['yTest'] = yTestList[fold-1]
	test_json = testDF.to_dict(orient='records')
	return jsonify(test_json)

# takes in selectedFeat and concatenates results to inputGraphResults dataframe 
@app.route('/inputGraphs/<selectedFeats>')
def inputGraphs(selectedFeats):
	selectedFeats = selectedFeats.split(",")
	numFeat = len(selectedFeats)
	selectedFeatDF = pd.DataFrame()
	# creates correctly structured string of features selected
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
	inputGraphResults_json = inputGraphResults.to_dict(orient='records')
	return jsonify(inputGraphResults_json)

# returns the lowest MSE for each possible number of features
@app.route('/lowestMSEByFeatureCount')
def lowestMSEByFeatureCount():
	results = pd.DataFrame()
    # append lists to results dataframe 
	results['numFeat'] = numFeatList
	results['mse'] = mseListLowest
	results['features'] = featuresList
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

# linear results for all scaled feature variables 
@app.route('/linearResults')
def linearResults():
	reg.fit(XScaled, y)
	results = getResultsDF(reg, np.array(XScaled.columns))
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

# returns regression results for number of features to include
@app.route('/featureSelectionResults/<numFeat>')
def featureSelectionResults(numFeat):
	numFeat = int(numFeat)
	results = pd.DataFrame()
	# append lists to results dataframe 
	results['numFeat'] = numFeatList
	results['mse'] = mseListLowest
	results['features'] = featuresList
	numFeatRow = results.loc[results['numFeat'] == numFeat]
	# get string of feature names 
	featNamesStr = ''
	for elem in numFeatRow['features']:
		featNamesStr+=elem
    # list of feature names 
	featNames = featNamesStr.split(',')
	# trim spaces off names
	i = 0
	while i < len(featNames):
		feat = featNames[i]
		featNames[i] = feat.strip()
		i +=1
    # get data for each feature and run regression 
	XNumFeat = XScaled[featNames]
	reg.fit(XNumFeat,y)
	results = getResultsDF(reg, featNames)
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

# lasso results for given lambda value
@app.route('/lassoResults/<_lambda>')
def lassoResults(_lambda):
	_lambda = float(_lambda)
	lasso = Lasso(alpha=_lambda)
	lasso.fit(XScaled, y)
	results = getResultsDF(lasso, np.array(XScaled.columns))
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

# ridge results for given lambda value
@app.route('/ridgeResults/<_lambda>')
def ridgeResults(_lambda):
	_lambda = float(_lambda)
	ridge = Ridge(alpha=_lambda)
	ridge.fit(XScaled, y)
	results = getResultsDF(ridge, np.array(XScaled.columns))
	results_json = results.to_dict(orient='records')
	return jsonify(results_json)

if __name__ == '__main__':
    app.run(debug=True)