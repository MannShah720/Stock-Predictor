# Apple Stock Predictor

## What I've Learned
- I use support vector regression, which is a supervised ML algorithm that finds a decision boundary that fits within a specific margin of error and is used to predict continuous values (such as stock price)

- When using an SVR, we want to achieve two outcomes:
  1. A line with the largest minimum margin (i.e. Pick the line such that the distance between the line and the closest point is largest)
  2. A line that correctly separates as many instances as possible

- The penalty parameter `C = 1e3` controls how strongly the SVR model tries to fit the training data and therefore is used to determine the 2nd outcome

- I use 3 types of SVR:
  1. Linear SVR: Assumes the relationship is a straight line.
  2. Polynomial SVR: With `degree` of 2, assumes a quadratic relationship.
  3. Radial Basis Function: Lets the data determine the shape of the curve and the `gamma` parameter controls how far the influence of each training data point reaches.
