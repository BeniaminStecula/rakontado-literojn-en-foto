#!/usr/bin/env python3.5
"""Contains classes for computations on data describing iris flowers."""

from decimal                import Decimal

from scipy.spatial.distance import euclidean


def arequal_icase(a, b):
	"""Check if strings a, b are equal."""
	try:
		return a.lower() == b.lower()
	except AttributeError:
		return a == b


class iris:
	"""Holds iris mesaurments and its class.

	Static fields:
		header  - list, holds name of each field
		row_len - record length (5)
		classes	- list holding iris class names; it is built as new records (with new class names) are added

	Fields (and their keys):
		sepal_length (0, 'sepal length)
		sepal_width  (1, 'sepal width')
		petal_length (2, 'petal length')
		petal_width  (3, 'peatal width')
		class_       (4, 'class')
	Keys are case insensitive.
	Length / width fields contain numbers of type Decimal (from decimal module).
	"""

	header       = ["sepal length", "sepal width", "petal length", "petal width", "class"]
	row_len = 5
	classes = []

	def set_class (self, irisclass):
		"""Helper function for setting class field"""
		try:
			self.class_ = iris.classes.index (irisclass)
		except ValueError:
			iris.classes.append (irisclass)
			self.class_ = iris.classes.__len__() - 1

	def __init__(self, record):
		"""Constructs iris record.

	Accpets string as input: "{nubmer},{number},{number},{number},{string}"

	Raises Exception with 3 arguments in case input line holds wrong number of fields:
		message
		record (after splitting)
		record length
		"""
		record = record.split (',')
		if record.__len__() != 5:
			raise Exception ("Wrong field amount in input string.", record, record.__len__())
		self.sepal_length = Decimal (record[0])
		self.sepal_width  = Decimal (record[1])
		self.petal_length = Decimal (record[2])
		self.petal_width  = Decimal (record[3])
		self.set_class (record[4])

	def __str__(self):
		"""Returns iris record as csv file line."""
		return "{0},{1},{2},{3},{4}".format(self.sepal_length, self.sepal_width, self.petal_length, self.petal_width, iris.classes [self.class_])

	def __getitem__(self, key):
		"""Raises KeyError in case not listed value is passed.
	See class description for list of avaiable keys.
		"""
		if isinstance (key, str):
			try:
				key = iris.header.index (key)
			except ValueError:
				raise KeyError (key)
		if   key == 0:
			return self.sepal_length
		elif key == 1:
			return self.sepal_width
		elif key == 2:
			return self.petal_length
		elif key == 3:
			return self.petal_width
		elif key == 4:
			return iris.classes[self.class_]
		else:
			raise KeyError (key)

	def __setitem__(self, key, val):
		"""	Raises KeyError in case not listed value is passed.
	See class description for list of avaiable keys.
		"""
		if isinstance (key, str):
			try:
				key = iris.header.index (key)
			except ValueError:
				raise KeyError (key)
		if   key == 0:
			self.sepal_length = Decimal (val)
		elif key == 1:
			self.sepal_width = Decimal (val)
		elif key == 2:
			self.petal_length = Decimal (val)
		elif key == 3:
			self.petal_width = Decimal (val)
		elif key == 4:
			set_class (val)
		else:
			raise KeyError (key)

	@property
	def get_vector (self):
		"""Get iris numerical data as list:
	[sepal_length, sepal_width, petal_length, petal_width]
		"""
		return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]

	def eucliD_to (self, other):
		"""Get euclidean distance from self to chosen iris record.

	Parameter:
		other - another iris object
		"""
		return euclidean (self.get_vector, other.get_vector)


class iris_data:
	"""Container for iris records.

	fields:
		data        - list of iris objects
		data_amount - data list length
	"""

	def __init__(self, data_path):
		self.data = []
		for d in open(data_path, 'r').read().split('\n'):
			if d == "":
				continue
			self.data.append (iris (d))
		self.data_amount = self.data.__len__()			

	def __getitem__(self, key):
		return self.data[key]
	def __setitem__(self, key, val):
		self.data[key] = val

	def getCSV(self):
		"""Converts contained iris data into CSV file data."""
		result = ""
		for d in self.data:
			result += str(d) + '\n'
		return result


class iris_data_iter:
	"""For list of iris objects, allows iteration over their fields.

	iris_data_obj - of type iris_data
	col           - key representing iris object field
	                see iris class description for list of avaiable keys

	In case iris_data_obj argument is of wrong type
	TypeError is raised with dictionary containing:
		'received' type
		'expected' type
	"""
	def __init__(self, iris_data_obj, col):
		if isinstance (iris_data_obj, iris_data):
			self.iris_data_obj = iris_data_obj
			self.col = col
			self.i = 0
		else:	
			raise TypeError ({'received': type (iris_data_obj), 'expected': iris_data})

	def __iter__(self):
		return self

	def __next__(self):
		if self.i < self.iris_data_obj.data_amount:
			self.i += 1
			return self.iris_data_obj.data[self.i-1][self.col]
		else:
			raise StopIteration

