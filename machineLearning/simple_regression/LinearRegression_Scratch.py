#y= mx + b rules here
#m = gradient, b = y-intercept

#(two-deminsional)
#m finding mean for best-fit 
#m = ( (mean(x) * mean(y)) - mean(x*y) ) / ( (mean(x))**2) - mean((x)**2) )

#finding y-intercept (best-fit)
#b = mean(y) - m * ( mean(x) )


from statistics import mean
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import random

style.use('fivethirtyeight')


#xs = np.array([i for i in range(1,7)], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    #hm is how many datapoint we want to create
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation== 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
        xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys ,np.float64)



def best_fit_slope_and_intercept(xs, ys):
    m = (  ( (mean(xs) * mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean((xs**2))))
    b = mean(ys) - m*mean(xs)
    return m, b


#squared error would be (generated output - actual output)^2 

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) **2)


#Finding r^2 is equal to 1 - ( SE(generated output))  / ( SE(mean of the actual output) ))

def coeffieciet_of_determination(ys_orig, ys_line):
    ys_mean_line = [mean(ys_orig) for y in ys_orig]
    sqaured_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, ys_mean_line)
    return 1 - (sqaured_error_regr / squared_error_mean)



xs, ys = create_dataset(40, 80,2, correlation='pos') 


m,b = best_fit_slope_and_intercept(xs, ys)




regression_line = [(m*x) + b for x in xs] # regression_line is y as y=mx+b, that's what the [loop] is doing-> mx + b

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coeffieciet_of_determination(ys, regression_line)

print(r_squared)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='r')
plt.plot(xs,regression_line)
plt.show()