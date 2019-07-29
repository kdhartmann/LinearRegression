import json
from flask import jsonify
import pytest
from app.lrapi import app


class TestClass(object):
	
	def setup_method(self):

		self.app = app
		self.client = self.app.test_client()

		return

	def test_nothing(self):
		r = self.client.get('/')
		r_json = r.get_json()
		assert r_json['message'] == 'This is an api.'

		return

	def test_linear_regression(self):

		with self.app.app_context():
			r = self.client.post(
				'/linear_regression',
				query_string={'scale': 'scaled'},
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

		r = self.client.get('/rooms', query_string={'scale': 'scaled'})
		r_json = r.get_json()

		assert len(r_json) > 0
		assert 'rooms' in r_json[0]

	def teardown_method(self):
		return

