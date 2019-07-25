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

def get_results_df_sort(model, features):
	"""Inputs: model (model used to fit regression) and features (list of names of features in model)
	Creates a dataframe for feature name and coefficient associated with that feature
	including the intercept; does sort by coefficient value
	Output: dataframe consisting of 'feature' and 'coef'.
	"""
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

def get_results_df(model, features):
	"""Inputs: model (model used to fit regression) and features (list of names of features in model)
	Creates a dataframe for feature name and coefficient associated with that feature
	including the intercept; does not sort by coefficient value
	Output: dataframe consisting of 'feature' and 'coef'.
	"""
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

def find_next_feat(included, excluded):
	"""Inputs: included (features already in model) and excluded (features not in model)
	Looks through all the excluded features and finds the one that will create the lowest
	mse when it is added to the feautures already included in the model
	Outputs: feature that produces lowest MSE and that lowest MSE value
	"""
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



## APIs

@app.route('/linear_regression', methods=['POST'])
def linear_regression():
	"""Inputs: scale (scaled or unscaled) and a json consisting of features to include in regression 
	Runs a linear regression and creates a dataframe of the feature name and its coefficient 
	Output: json consiting of 'feature' and 'coef'.
	"""
	feature_names = request.get_json()
	if request.args['scale']=='scaled':
		feature_matrix_df = XScaled[feature_names]
	elif request.args['scale']=='unscaled':
		feature_matrix_df = X[feature_names]
	reg.fit(feature_matrix_df, y)
	return jsonify(
		get_results_df(reg, feature_names).to_dict(orient='records')
	)


@app.route('/rooms', methods=['GET'])
def rooms():
	"""Input: scale (scaled or unscaled)
	Creates a dataframe of the scaled or unscaled verison of rooms
	Output: json consisting of 'rooms'.
	"""
	if request.args['scale']=='scaled':
		df = XScaled
	elif request.args['scale']=='unscaled':
		df = X
	return jsonify(
		df[['rooms']].to_dict(orient='records')
	)


@app.route('/kfold_mse', methods=['GET'])
def kfoldmse():
	"""Input: none
	Creates a dataframe of the mse for each fold in k-fold
	Output: json consisting of 'fold' and 'mse'.
	"""
	results = pd.DataFrame({
		'fold': split_list,
		'mse': mse_list
	})
	return jsonify(
		results.to_dict(orient='records')
	)

@app.route('/kfold_sets', methods=['GET'])
def kfold_sets():
	"""Inputs: fold_num (1, 2, 3, 4, or 5) and fold_set (train or test)
	Creates a dataframe of the X and y train or test values for a specified fold
	Output: json consiting of 'XTrain' and 'yTrain' or 'XTest' and 'yTest'
	"""
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

@app.route('/lowest_mse_by_count', methods=['GET'])
def lowest_mse_by_count():
	"""Input: none
	Calls on dataframe that holds the lowest MSE for each number of features and the features 
	included; elements in 'features' are previously formatted into a string and not a list
	Output: json consisting of 'numFeat' 'mse' and 'features'.
	"""
	return jsonify(
		lowest_mse_df.to_dict(orient='records')
	)

@app.route('/linear_regression_all', methods=['GET'])
def linearResults():
	"""Input: none
	Runs a regression with all scaled features and creates dataframe with feature and coefficient
	Output: json consisting of 'feature' and 'coef'.
	"""
	reg.fit(XScaled, y)
	results = get_results_df_sort(reg, np.array(XScaled.columns))
	return jsonify(
		results.to_dict(orient='records')
	)

@app.route('/feature_selection_results', methods=['GET'])
def feature_selection_results():
	"""Input: num_feats (number of features to include in model: 1-8)
	Calls on the lowest_mse_df, turns 'features' from a string to a list, runs 
	regression with those features, and creates dataframe with feature and coefficient 
	Output: json consisting of 'feature' and 'coef'.
	"""
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

@app.route('/lasso_results', methods=['GET'])
def lasso_results():
	"""Input: alpha (alpha value for lasso regression)
	Runs a lasso regression with given alpha and creates dataframe of feature name and coefficient
	Ouput: json consisting of 'feature' and 'coef'.
	"""
	lasso = Lasso(alpha=float(request.args['alpha']))
	lasso.fit(XScaled,y)
	return jsonify(
		get_results_df_sort(lasso, np.array(XScaled.columns)).to_dict(orient='records')
	)

@app.route('input_graphs', methods=['POST'])
def input_graphs():
	"""Input: json consisting of features to include in regression
	Runs linear regression with features provided, calculates mse, and converts feature names
	into a string; elements in 'selectedFeat' are formatted to be a string
	Output: json consisting of 'numFeat' 'mse' and 'selectedFeat'
	"""
	feature_names = request.get_json()
	feature_string = ''
	for elem in feature_names:
		if len(feature_string) == 0:
			feature_string += elem
		else:
			feature_string += (f", {elem}")
	df = pd.DataFrame({
		'numFeat': [len(feature_names)],
		'mse': [np.mean((cross_val_score(reg, XScaled[feature_names], y, cv=cv, scoring='neg_mean_squared_error'))*-1)],
		'selectedFeat': [feature_string]
	})
	return jsonify(
		df.to_dict(orient='records')
	)

if __name__ == '__main__':
    app.run(debug=True)