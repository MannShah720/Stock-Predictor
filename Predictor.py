import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt



dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('/')[0]))
			prices.append(float(row[1]))
	return

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

    # SVR models
	svr_lin = SVR(kernel= 'linear', C= 1e3)            # Linear Kernel
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)  # Polynomial Kernel
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)   # Radial Basis Function Kernel

    # Fitting models
	svr_rbf.fit(dates, prices)
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)

    # Plotting the results
	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting initial datapoints 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model')
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') 
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model')
	
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('appl.csv')

predicted_price = predict_price(dates, prices, 30)  