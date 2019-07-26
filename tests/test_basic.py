import pytest
from app.lrapi import app


class TestClass(object):
	
	def setup_method(self):

		self.client = app.test_client()

		return

	def test_nothing(self):
		assert 1==1
		return
	
	def teardown_method(self):
		return

