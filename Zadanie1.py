#!/usr/bin/env python3.5

# TODO: tasks 6 - ...
# TODO: consider making function to plot / scatter stuff in order to eliminate code repetition

import sys
# exit

import matplotlib.pyplot as plt
# subplots, show


from decimal           import Decimal
from statistics        import mean, stdev

from numpy             import fromiter
from matplotlib.colors import hsv_to_rgb

from iris              import iris, iris_data, iris_data_iter

def main():
	"""Task solutions."""

	dane = iris_data ("iris.csv")

	# 1
	print ("1.\nLiczba próbek: {0}\nIlość atrybutów: {1}\n".format (dane.data_amount, iris.row_len))

	# 2
	print ("2.\nPróbka nr 10: {0}\nPróbka nr 75: {1}\nOdległość Euklidesowa: {2}\n".format (dane[9], dane[74], dane[9].eucliD_to (dane[74])))

	# 3
	print ("3.")
	daneIterator = iris_data_iter (dane, 0)
	daneObliczenia = dict ()
	for col in iris.header[:4]:
		danaObliczenia = {}
		print ("%s: " % col)
		for lab, func in [('min', min), ('max', max), ('mean', mean), ('stdev', stdev)]:
			daneIterator.i   = 0
			daneIterator.col = col
			danaObliczenia[lab] = func (daneIterator)
			print ("\t%s = %s" % (lab, danaObliczenia[lab]))
		daneObliczenia[col] = danaObliczenia
	print ()

	# 4
	fig_zd4, ax_zd4 = plt.subplots ()
	fig_zd4.suptitle ("Zadanie 4")
	ax_zd4.scatter (fromiter (iris_data_iter (dane, 0), float), fromiter (iris_data_iter (dane, 1), float))
	ax_zd4.set_xlabel ('Sepal length')
	ax_zd4.set_ylabel ('Sepal width')
	ax_zd4.grid (True)
	plt.show ()

	# 5
	fig_zd5, ax_zd5 = plt.subplots ()
	fig_zd5.suptitle ("Zadanie 5")
	
	hues = {}
	hueBase = float(iris.classes.__len__()) ** -1.
	i=0
	while i < iris.classes.__len__():
		hues[iris.classes[i]] = i * hueBase
		i += 1
	for dana in dane:
		ax_zd5.scatter (dana[0], dana[2], color = hsv_to_rgb ([hues[dana[4]], 1, 0.9]))
	ax_zd5.set_xlabel ('Sepal length')
	ax_zd5.set_ylabel ('Petal length')
	ax_zd5.grid (True)
	plt.show ()


if __name__ == "__main__":
	sys.exit(main())

