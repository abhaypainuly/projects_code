#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
import pandas as pd


# In[ ]:


#Predicting output with one input
#Linear Regression in 2D


# In[350]:


x = [1,4,43,34,32,52,26,52,45,25]
y = [2,24,51,35,6,15,35,63,36,8]
print(len(x)==len(y))


# In[433]:


#Initializing random Weights and Bias
w = np.random.randn()
b = np.random.randn()
print('Initial//Random parameters:\nWeight: {}\nBias: {}'.format(w,b))

dit = {}
costs = []
learning_factor = 0.0000001

for itr in range(200000):    
    i = np.random.randint(len(x))
    if i not in dit:
        dit[i] = 0
    dit[i] += 1
    
    #Predicting the output
    predict = w*x[i]+b

    #Calculating difference with output
    #cost = np.square(predict-y[i])
    if itr%1000 == 0:
        cost_sum = 0
        for i in range(len(x)):
            cost_sum += np.square(w*x[i]+b-y[i])
        costs.append(cost_sum)    

    #Calculating partial derivative
    dcost_dw = 2*(predict-y[i])*x[i]
    dcost_db = 2*(predict-y[i])*1

    #Updating the weights and bias
    w = w - learning_factor*dcost_dw
    b = b - learninig_factor*dcost_db

print('\nTrained parameters:\nWeight: {}\nBias: {}'.format(w,b))
print(dit)


# In[434]:


plt.grid()
plt.title('Change in Cost function!')
fig = plt.plot(costs)
plt.show()


# In[435]:


plt.title('Linear Regression model!',fontsize=10)

#Ploting predicted linear regression model
x_axis = [i for i in range(70)]
y_axis = [w*i+b for i in x_axis]
plt.plot(x_axis,y_axis,color='Red')

#Ploting the points
plt.scatter(x,y,color='Blue')

#Displaying the graph
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Predicting Color of Flower using two inputs
#Linear Regression in 3D


# In[194]:


data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

mystery_flower = [4.5, 1]


# In[28]:


#Activation function
def sigmoid(x):    
    return 1/(1+np.exp(-x))


# In[29]:


#Derivative of activation function
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))


# In[332]:


#Converting prediction to color 
def color_p(x):
    if x<=.5:
        return 'Blue'
    return 'Red'


# In[345]:


#Visualizing Sigmoid function and its Derivative
X = np.linspace(-5, 5, 100)
plt.plot(X, sigmoid(X), c="Blue")
fig = plt.plot(X, sigmoid_p(X), c="Red")


# In[322]:


costs = []
learning_rate = 0.1

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

for itr in range(60000):    
    #Selecting random training data
    i = np.random.randint(len(data))    
    
    point = data[i]
    target = data[i][2]

    #Calculating Weighted Sum and Cost
    z = w1*point[0] + w2*point[1] + b
    prediction = sigmoid(z)
    cost = np.square(prediction-target)
    
    #Storing costs values to plot a graph
    if i % 100 == 0:
        c = 0
        for j in range(len(data)):
            p = data[j]
            p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
            c += np.square(p_pred - p[2])
        costs.append(c)

    #Calculating partial derivative using chain rule
    dcost_dz = 2*(prediction-target)*sigmoid_p(z)
    
    dcost_dw1 = dcost_dz*point[0]
    dcost_dw2 = dcost_dz*point[1]
    dcost_db = dcost_dz*1

    #Updating the weights
    w1 = w1 - learning_rate*dcost_dw1
    w2 = w2 - learning_rate*dcost_dw2
    b = b - learning_rate*dcost_db


# In[323]:


plt.grid()
fig = plt.plot(costs)
plt.show()


# In[335]:


#Prediction for training inputs
for i in range(len(data)):
    pred = sigmoid(w1*data[i][0] + w2 *data[i][1] + b)
    color = color_p(pred)
    print(data[i])
    print(pred,color)
    plt.scatter(data[i][0],data[i][1],c=color)


# In[334]:


#Prediction for Mystery flower
print(color_p(sigmoid(w1*mystery_flower[0]+w2*mystery_flower[1]+b)))


# In[347]:


plt.grid()
#Networks predictions in the x,y plane
for x in np.linspace(0, 6, 20): #Ploting the Grid into which color they fall
    for y in np.linspace(0, 3, 20):
        pred = sigmoid(w1 * x + w2 * y + b)        
        plt.scatter([x],[y],c=color_p(pred), alpha=.2)


for val in data: #Ploting the training data
    pred = sigmoid(w1 * val[0] + w2 * val[1] + b)
    plt.scatter(val[0],val[1],c=color_p(pred))

#Ploting the prdicted flower with 'Grey'
plt.scatter(mystery_flower[0],mystery_flower[1],c='grey)


# In[348]:


#Printing Linear Regression Model
x_axis = np.linspace(0, 6, 20)
y_axis = np.linspace(0, 6, 20)

f = lambda x1,x2:w1*x1+w2*x2+b

X, Y = np.meshgrid(x_axis, y_axis)
Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# In[ ]:





# In[ ]:


#Testing model with iris dataset


# In[281]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


# In[282]:


#Loading the data
boston = load_boston()

#Spliting the data into training and testing
x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.2)

x_train = np.array(x_train)
y_train = np.array(y_train)

print(boston.feature_names)


# In[283]:


x_train.shape


# In[284]:


y_train.shape


# In[285]:


boston.data.shape


# In[286]:


x_data = pd.DataFrame(x_train)
x_data.columns = boston.feature_names
x_data.describe()


# In[287]:


#x_data.drop(['NOX'], axis=1, inplace = True)
for x in x_data.columns:
    '''if x in ['CHAS','NOX',]:
        continue'''
    x_data[x] = (x_data[x] - x_data[x].mean())/x_data[x].std()
x_train = x_data.to_numpy()
x_data.describe()


# In[316]:


#Generating random weights and bias
np.random.seed(2)
weights = 2*np.random.random((13,1))-1
bias = np.random.randn()

#List to keep track of cost function
costs = []
iterations = 200000

#Learning_rate
alpha = 0.0001

for itr in range(iterations):
    i = np.random.randint(len(x_train))
    point = x_train[i]
    target = y_train[i]
    
    #Predicting the output
    prediction = point.dot(weights) + bias
    
    #Calcutaing cost function
    #cost = (predicton-target)**2
    
    #Storing values of cost for visualisation purpose   
    if i % 1000 == 0:
        c = 0
        for j in range(len(x_train)):
            x = x_train[j]
            predict = x.dot(weights) + bias
            c += np.square(predict - y_train[j])
        costs.append(c/len(x_train))
    
    #Backprogation to calculate adjustments
    dcost_dz = 2*(prediction-target)
    
    dcost_dw = dcost_dz*point.T
    dcost_dw = np.array([dcost_dw])    
    dcost_dz = dcost_dz*1
    
    #Updating the weights and bias
    weights = weights - alpha*dcost_dw.T
    bias = bias - alpha*dcost_dz


# In[317]:


#Ploting cost
plt.grid()
fig = plt.plot(costs)
plt.show()


# In[318]:


x_data = pd.DataFrame(x_test)
x_data.columns = boston.feature_names
#x_data.drop(['NOX'], axis=1, inplace = True)
for x in x_data.columns:
    x_data[x] = (x_data[x] - x_data[x].mean())/x_data[x].std()
x_test = x_data.to_numpy()

for i in range(len(x_test)):
    '''if x in ['CHAS','NOX',]:
        continue'''
    pred = x_test[i].dot(weights) + bias
    print(y_test[i],pred,y_test[i]-pred)


# In[319]:


print(costs[-1],min(costs))
print(costs[-1]**0.5,min(costs)**0.5)

