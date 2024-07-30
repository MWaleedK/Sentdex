#y= mx + b rules here
#m = gradient, b = y-intercept

#(two-deminsional)
#m finding mean for best-fit 
#m = ( (mean(x) * mean(y)) - mean(x*y) ) / ( (mean(x))**2) - mean((x)**2) )

#finding y-intercept (best-fit)
#b = mean(y) - m * ( mean(x) )


from statistics import mean
import numpy as np
from matplotlib.pyplot import style
import matplotlib.pyplot as plt

xs = np.array([i for i in range(1,7)], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (  ( (mean(xs) * mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean((xs**2))))
    b = mean(ys) - m*mean(xs)
    return m, b

m,b = best_fit_slope_and_intercept(xs, ys)




regression_line = [(m*x) + b for x in xs] # regression_line is y as y=mx+b, that's what the [loop] is doing-> mx + b

predict_x = 8
predict_y = (m*predict_x) + b


plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs,regression_line)
plt.show()