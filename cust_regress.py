# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import random
from statistics import mean
from matplotlib import style

style.use('ggplot')

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

# regression model
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

# determine squared error
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)*(ys_line - ys_orig))

#coefficient of determination (R squared)
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

#creating dataset
def create_dataset(hm,variance, step=2,correlation=False):
    """create_dataset

    :param hm: how much = how big is data set
    :param variance: range of values (postitive and negative, centered from an incremental value val
    :param step: step is added for every increment in val
    :param correlation: 'pos' or 'neg' (positive or negative)
    """
    val =1
    ys=[]
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)

#create a data set
xs, ys = create_dataset(40, 10, 2, correlation='neg')

m, b = best_fit_slope_and_intercept(xs,ys)
print('best slope coeffcient=',m,'best intercept=',b)

regression_line = [(m*x)+b for x in xs]
# plot a function based on the prediction of model
r_squared = coefficient_of_determination(ys,regression_line)
print('given line variables, coeffcient of determination=',r_squared)

plt.scatter(xs,ys,color='#003F72',label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()

