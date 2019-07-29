import json
from flask import jsonify
import pytest
from app.lrapi import app


class TestClass(object):
	
	def setup_method(self):
		self.app = app
		self.client = self.app.test_client()
		return


	def test_linear_regression(self):
		params = ['scaled', 'unscaled']
		for elem in params: 
			with self.app.app_context():
				r = self.client.post(
					'/linear_regression',
					query_string={'scale': elem},
					data=json.dumps({'features': ['age', 'rooms']}),
					headers={'Content-Type': 'application/json'}
				)
				r_json = r.get_json()

				assert 'feature' in r_json[0]
				assert 'coef' in r_json[0]

				features = [e['feature'] for e in r_json]

				assert 'intercept' in features
				assert 'age' in features
				assert 'rooms' in features
				assert not 'lotSize' in features


	def test_rooms(self):
		params = ['scaled', 'unscaled']

		for elem in params:
			r = self.client.get('/rooms', query_string={'scale': elem})
			r_json = r.get_json()

			assert len(r_json) > 0
			assert 'rooms' in r_json[0]


	def test_kfold_mse(self):
		r = self.client.get('kfold_mse')
		r_json = r.get_json()

		assert len(r_json) == 5
		assert 'fold' in r_json[0]
		assert 'mse' in r_json[0]


	def test_kfold_sets_train(self):
		r = self.client.get('/kfold_sets', query_string={'fold_num': 1, 'fold_set': 'train'})
		r_json = r.get_json()

		assert 'XTrain' in r_json[0]
		assert 'yTrain' in r_json[0]
		assert 'XTest' not in r_json[0]
		assert 'yTest' not in r_json[0]


	def test_kfold_sets_test(self):
		r = self.client.get('/kfold_sets', query_string={'fold_num': 1, 'fold_set': 'test'})
		r_json = r.get_json()

		assert 'XTrain' not in r_json[0]
		assert 'yTrain' not in r_json[0]
		assert 'XTest' in r_json[0]
		assert 'yTest' in r_json[0]


	def test_lowest_mse_by_count(self):
		r = self.client.get('/lowest_mse_by_count')
		r_json = r.get_json()

		assert 'numFeat' in r_json[0]
		assert 'mse' in r_json[0]
		assert 'features' in r_json[0]
		assert len(r_json) == 8

		features = [e['features'] for e in r_json]
		assert type(features[0]) == str


	def test_linear_regression_all(self):
		r = self.client.get('/linear_regression_all')
		r_json = r.get_json()

		assert len(r_json) == 9
		assert 'feature' in r_json[0]
		assert 'coef' in r_json[0]

		features = [e['feature'] for e in r_json]
		assert 'intercept' in features


	def test_feature_selection_results(self):
		r = self.client.get('/feature_selection_results', query_string={'num_feats': 5})
		r_json = r.get_json()

		assert 'feature' in r_json[0]
		assert 'coef' in r_json[0]
		assert len(r_json) == 6

		features = [e['feature'] for e in r_json]
		assert 'intercept' in features


	def test_lasso_results(self):
		r = self.client.get('/lasso_results', query_string={'alpha': .0001})
		r_json= r.get_json()

		assert 'feature' in r_json[0]
		assert 'coef' in r_json[0]
		len(r_json) == 9


	def test_input_graphs(self):
		with self.app.app_context():
			r = self.client.post(
				'/input_graphs',
				data=json.dumps({'features': ['age', 'rooms']}),
				headers={'Content-Type': 'application/json'}
			)
			r_json = r.get_json()

			assert 'numFeat' in r_json[0]
			assert 'mse' in r_json[0]
			assert 'selectedFeat' in r_json[0]

			num_feats = [e['numFeat'] for e in r_json]
			assert num_feats[0] == 2


	def teardown_method(self):
		return

