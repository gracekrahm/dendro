import csv
import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.patches
from astropy.io import fits
from astropy.stats import mad_std
from astropy import stats
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.integrate import simps
import h5py
from astrodendro import Dendrogram
import peakutils.peak
import datetime
import multiprocessing as mp
import sys
from scipy.optimize import fsolve
from sympy import Eq, Symbol, solve, symbols
import warnings 
import scipy.constants as C
import pandas as pd

import warnings
warnings.simplefilter('ignore')

def fract_fit(A,D):
	return A**(D/2)


if __name__ == "__main__":

	propfile13 = '13CO21_clump_properties_npix_area_full.csv'
	propfile12 = '12CO21_clump_props_final.csv'
	df12 = pd.read_csv(propfile12)
	area12 = df12['area']
	perim12 = df12['perimeter']


	df13 = pd.read_csv(propfile13)
	df13 = df13.drop([17], axis = 0)
	area_column13 = df13['area']
	area_column_13 = df13['area npix']
#	print(df[df.area >= 60000])
	perim_column13 = df13['perimeter']	
#	print(df13[df13.perimeter >= 3963])	
	
	xdata13 = area_column13
	ydata13 = perim_column13
	popt13, pcov13 = curve_fit(fract_fit, xdata13, ydata13)
	print(popt13)

	popt12, pcov12 = curve_fit(fract_fit, area12, perim12)
	print(popt12)
#	print(np.nanmax(xdata13))
#	print(np.nanmax(ydata13))
	plt.scatter(xdata13, ydata13, label = '13CO', color = 'red')
	plt.scatter(area12, perim12, label = '12CO', color = 'blue')
	model_range = list(range(0,75018))
	plt.plot(model_range, fract_fit(model_range, popt13), label='13CO, D₂='+str(*popt13), color='red')	
	plt.plot(model_range, fract_fit(model_range, popt12), label = '12CO, D₂='+str(*popt12), color = 'blue')

	plt.xlim([0,21242])
	plt.ylim([0,2000])
	plt.legend(loc='lower right')
	plt.xlabel('Area (pc^2)')
	plt.ylabel('Perimeter (pc)')
	plt.title('D₂')
	plt.show()

	print("done")

