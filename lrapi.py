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
from sklearn.feature_selection import f_regression


app = Flask(__name__)
CORS(app)

# function to get dataframe of regression results: coefficients and features with sorting 
def get_results_df_sort(model, features):
    df = pd.DataFrame()
    names = ['intercept']
    coefs = [round(model.intercept_,2)]
    for elem in features:
        names.append(elem)
    for elem in model.coef_:
        coefs.append(round(elem,2))
    df['feature'] = names
    df['coef'] = coefs
    return df.reindex(
    	df.coef.abs().sort_values(ascending = False).index
    )

# function to get dataframe of regression results: coefficients and features without sorting 
def get_results_df(model, features):
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

# finds the next best feature in excluded to add to included
def find_next_feat(included, excluded):
    mse_list = []
    global XScaled
    for elem in excluded:
        feats_list = included.tolist()
        feats_list.append(elem)
        mse_list.append(
        	np.mean((cross_val_score(reg, XScaled[feats_list], y, cv=cv, scoring='neg_mean_squared_error'))*-1)
        )
    lowest_index = mse_list.index(min(mse_list))
    return excluded[lowest_index], min(mse_list)

## pre-processing
dataset = pd.read_csv('SaratogaHousesClean.csv')
y = dataset['price']
X = dataset.iloc[:, 1:9]

scaler = StandardScaler()
cv = KFold(n_splits=5, shuffle=False)
reg = LinearRegression(fit_intercept = True)

# scaling the features 
XScaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# use in price = livingArea model for CV
livingArea = np.array(XScaled['livingArea']).reshape(-1,1)

## K-fold: get x/y train/test points and their MSE - used in kfold_mse and kfold_sets
mse_list = []
split_list = []
XTrain_list = []
yTrain_list = []
XTest_list = []
yTest_list = []
split = 1

# loop through the splits 
for train_index, test_index in cv.split(livingArea):
	X_train, X_test, y_train, y_test = livingArea[train_index], livingArea[test_index], y[train_index], y[test_index]
	reg.fit(X_train, y_train)
	mse_list.append(
		metrics.mean_squared_error(y_test, reg.predict(X_test))
	)
	# format the X train and test correctly 
	X_train_correct = []
	for elem in X_train:
		X_train_correct.append(elem[0])
	X_test_correct = []
	for elem in X_test:
		X_test_correct.append(elem[0])
	# append results to list to later be added to dataframe
	XTrain_list.append(X_train_correct)
	yTrain_list.append(np.array(y_train))
	XTest_list.append(X_test_correct)
	yTest_list.append(np.array(y_test))
	split_list.append(split)
	split +=1

## Finds Lowest MSE for each Feature Number - used in lowest_mse_by_count and feature_selection_results
feature_num = 1
feats_included = []
features_list = []
mse_lowest_list = []
num_feat_list = []
potential_feats = XScaled.columns
# loops through each possible number of features
while feature_num <= 8:
	feats_included = np.asarray(feats_included)
	feats_excluded = np.setdiff1d(potential_feats, feats_included)
	# function returns next feature and the MSE
	feat_exported, lowest_mse = find_next_feat(feats_included, feats_excluded) 
	# add the next feature to the array of used features 
	feats_included = np.append(feats_included,feat_exported)
	# format list into string of features 
	feats_included_string = ''
	for elem in feats_included:
		if len(feats_included_string) == 0:
			feats_included_string += elem
		else:
			feats_included_string += (f", {elem}")
	# append results to lists 
	features_list.append(feats_included_string)
	mse_lowest_list.append(lowest_mse)
	num_feat_list.append(feats_included.shape[0])
	feature_num += 1

lowest_mse_df = pd.DataFrame({
	'numFeat': num_feat_list,
	'mse': mse_lowest_list,
	'features': features_list
})


# create the dataframe for the user input graph results 
inputGraphResults = pd.DataFrame(columns = ['model', 'numFeat', 'mse', 'selectedFeat'])


## APIs

# linear reg results for scaled or unscaled given the features
@app.route('/linear_regression', methods=['POST'])
def linear_regression():
	feature_names = request.get_json()
	if request.args['scale']=='scaled':
		feature_matrix_df = XScaled[feature_names]
	elif request.args['scale']=='unscaled':
		feature_matrix_df = X[feature_names]
	reg.fit(feature_matrix_df, y)
	return jsonify(
		get_results_df(reg, feature_names).to_dict(orient='records')
	)

# returns rooms for scaled or unscaled
@app.route('/rooms', methods=['GET'])
def rooms():
	if request.args['scale']=='scaled':
		df = XScaled
	elif request.args['scale']=='unscaled':
		df = X
	return jsonify(
		df[['rooms']].to_dict(orient='records')
	)

# returns dataframe of MSEs for all the folds that were calculated for mseList
@app.route('/kfold_mse', methods=['GET'])
def kfoldmse():
	results = pd.DataFrame({
		'fold': split_list,
		'mse': mse_list
	})
	return jsonify(
		results.to_dict(orient='records')
	)

# returns X and y of train or test sets
@app.route('/kfold_sets', methods=['GET'])
def kfold_sets():
	results = pd.DataFrame()
	fold_num = int(request.args['fold_num'])
	if request.args['fold_set']=='train':
		results['XTrain'] = XTrain_list[fold_num-1]
		results['yTrain'] = yTrain_list[fold_num-1]
	elif request.args['fold_set']=='test':
		results['XTest'] = XTest_list[fold_num-1]
		results['yTest'] = yTest_list[fold_num-1]
	return jsonify(
		results.to_dict(orient='records')
	)

# returns the lowest MSE for each possible number of features
@app.route('/lowest_mse_by_count', methods=['GET'])
def lowest_mse_by_count():
	return jsonify(
		lowest_mse_df.to_dict(orient='records')
	)

# linear results for all scaled feature variables 
@app.route('/linear_regression_all', methods=['GET'])
def linearResults():
	reg.fit(XScaled, y)
	results = get_results_df_sort(reg, np.array(XScaled.columns))
	return jsonify(
		results.to_dict(orient='records')
	)

# returns feature selection coefficients for given number of features
# uses lowest_mse_df and must break down the features (string) into a list
@app.route('/feature_selection_results', methods=['GET'])
def feature_selection_results():
	feature_row = lowest_mse_df.loc[lowest_mse_df['numFeat'] == int(request.args['num_feats'])]
	feature_name_string = ''
	for elem in feature_row['features']:
		feature_name_string += elem.strip()
	feature_name_string =''
	for elem in feature_row['features']:
		feature_name_string += elem
	features = []
	for elem in feature_name_string.split(','):
		features.append(elem.strip())
	reg.fit(XScaled[features], y)
	return jsonify(
    	get_results_df_sort(reg,feature_name_string.split(',')).to_dict(orient='records')
    )

# returns lasso coefficients for given alpha 
@app.route('/lasso_results', methods=['GET'])
def lasso_results():
	lasso = Lasso(alpha=float(request.args['alpha']))
	lasso.fit(XScaled,y)
	return jsonify(
		get_results_df_sort(lasso, np.array(XScaled.columns)).to_dict(orient='records')
	)

# NEED TO FIX BELOW THIS LINE

# takes in selectedFeat and concatenates results to inputGraphResults dataframe 
@app.route('/inputGraphs/<selectedFeats>')
def inputGraphs(selectedFeats):
	global inputGraphResults
	selectedFeats = selectedFeats.split(",")
	# creates correctly structured string of features selected
	outputString = ''
	for elem in selectedFeats:
		if len(outputString) == 0:
			outputString += elem
		else:
			outputString += (f", {elem}")
	# create df to concatenate to inputGraphResults
	toConcatDF = pd.DataFrame({
		'numFeat': [len(selectedFeats)],
		'mse': [np.mean((cross_val_score(reg, XScaled[selectedFeats], y, cv=cv, scoring='neg_mean_squared_error'))*-1)],
		'selectedFeat': [outputString],
		'model': [(inputGraphResults['selectedFeat'].shape[0] + 1)]
		})
	inputGraphResults = pd.concat([inputGraphResults, toConcatDF], sort = True)
	inputGraphResults_json = inputGraphResults.to_dict(orient='records')
	return jsonify(inputGraphResults_json)

if __name__ == '__main__':
    app.run(debug=True)