import csv
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


dates = []
prices = []


# Read data from CSV file
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # Skips the 1st row

        for row in csvFileReader:
            date = datetime.strptime(row[0], "%d/%m/%Y")  # Convert string to datetime object
            dates.append(date)
            prices.append(float(row[4]))  # Close price

    return


def predict_price(dates, prices, x):

    # =========== NumPy Operations ===========
    # Convert dates to the number of days since the first date
    first_date = dates[0]

    date_numbers = np.array([
        (date - first_date).days
        for date in dates
    ], dtype=float)

    date_numbers = date_numbers.reshape(-1, 1)  # Convert to 2D array for sklearn

    prices = np.array(prices)

    # Scaler learns the distribution and converts to standardized values
    scaler = StandardScaler()
    scaled_dates = scaler.fit_transform(date_numbers)

    # =========== SVR models ===========
    svr_lin = SVR( kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(scaled_dates, prices)
    svr_poly.fit(scaled_dates, prices)
    svr_rbf.fit(scaled_dates, prices)

    # Create smooth x-axis for plotting
    plot_dates = np.linspace(
        date_numbers.min(),
        date_numbers.max(),
        500
    ).reshape(-1, 1)

    scaled_plot_dates = scaler.transform(plot_dates)

    # Predictions
    rbf_predictions = svr_rbf.predict(scaled_plot_dates)
    lin_predictions = svr_lin.predict(scaled_plot_dates)
    poly_predictions = svr_poly.predict(scaled_plot_dates)

    # Convert numeric dates back to real dates for plotting
    plot_datetime = [
        first_date + np.timedelta64(int(day), 'D')
        for day in plot_dates.flatten()
    ]

    # =========== Plotting =========== 
    plt.figure(figsize=(12, 6))

    plt.scatter(dates, prices, color='black', s=15, label='Data')


    plt.plot(plot_datetime, rbf_predictions, color='red', label='RBF model')
    plt.plot(plot_datetime, lin_predictions, color='green', label='Linear model')
    plt.plot(plot_datetime, poly_predictions, color='blue', label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Apple Stock - Support Vector Regression')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # x is interpreted as number of days after first date
    x_scaled = scaler.transform([[x]])

    return (
        svr_rbf.predict(x_scaled)[0],
        svr_lin.predict(x_scaled)[0],
        svr_poly.predict(x_scaled)[0]
    )


get_data('appl.csv')

predicted_price = predict_price(dates, prices, 30)

print("RBF prediction:", f"${predicted_price[0]:.2f}")
print("Linear prediction:", f"${predicted_price[1]:.2f}")
print("Polynomial prediction:", f"${predicted_price[2]:.2f}")