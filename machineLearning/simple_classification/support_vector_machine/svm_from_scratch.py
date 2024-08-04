
import matplotlib.pyplot as plt
import cv2 as cv
import os

# remember finding the magnitude of a vector is : sqrt(x^2 = y2) (two dimensions)
# dot product between vectors is king here

# in a plane of (+)s and (-)s as class outputs to be classified as

#after the decision boundary is made, the svm will then going to classify the points by taking a point (say: W) that points perpendicularly to the hyperplane 
#use then find the unknow vector (say: U) 
# then project U on W or vice versa.
# check what side of the hyperplane it i on
# b is the bias, y-intercep in this case

# (U . W) + b >= 0 the classification output is +, otherwise -
# if (U . W) + b == 0, then the point is on the decision boundary

#We know the value for U since it is a vector and will have values as a vector does
#Our main concern is finding both W and b

#Support vectors are the feature sets that would move if a feature(point on plane is move) which is understandable

#why is Width necessary? because width = (SV_+ + SV_-)/2 will give the separation hyperplane where SV_+ and SV_- are support vectors or the margins from the separator hyperplane

#width is given by (X_+ - X_-) . (W/magnitude(W)), where we maximize width
# we want to minimize (1/magnitude(W)) and as well (2/magnitude(W))^2
# this is a constraint (2/magnitude(W))^2
#bring in largrange
'''
fig = plt.figure(figsize=(12,8))
image_base_path = os.path.dirname(os.path.abspath(__file__))

lagrange = os.path.join(image_base_path, 'lagrange.png')
lagrange = cv.imread(lagrange, cv.IMREAD_COLOR)
fig.add_subplot(2,2,1)
plt.imshow(lagrange)
plt.axis('off')
plt.title('Lagrange, maximize b')

#why maximize b? because it is the y-intercept, it moves the y-line up and down (on the y-axis). Call it bias in SVM when referring to a hyperplane
der_lagrange = os.path.join(image_base_path, 'lagrange_derivation.png')
#partital derivation done on lagrange with respect to W and b
der_lagrange = cv.imread(der_lagrange, cv.IMREAD_COLOR)
fig.add_subplot(2,2,2)
plt.imshow(der_lagrange)
plt.axis('off')
plt.title('Partial Derivation of lagrange')

#maximize this
lagrange_output = os.path.join(image_base_path, 'lagrange_output.png')
lagrange_output = cv.imread(lagrange_output, cv.IMREAD_COLOR)
fig.add_subplot(2,2, 3)
plt.imshow(lagrange_output)
plt.axis('off')
plt.title('Final output to be maximized')
plt.waitforbuttonpress()
'''


#Now svms do not scale well with a really big dataset since all feature sets are needed in memory
#Minibatches can be used
# But the most common method used instead of SVM is sequential minimal optimization or SMO
#But
#once trained on a feature set, you just need the sign (+/-) of WX+b (good point)

#it's an optimization proble, we want a global minimum

from matplotlib import style 
import numpy as np
import copy
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

        
        
        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))            

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()
        
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()