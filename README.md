# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required.
2.Read the dataset. 
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ROSHINI R K
RegisterNumber: 212222230123 
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data =np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
Y=data[:,2]

X[:5]

Y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)


def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
### Array value of x
![ML51](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/b98beff6-e934-435f-9ac5-c40979e18a78)

### Array Value of y
![ML52](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/d1044782-ef62-4094-9434-fe2da63d3338)

### Exam 1- Score graph
![ML53](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/e16a5e12-ee0f-44c2-ae40-4fea5ee2e5b9)

### Sigmoid Function Graph
![ML54](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/f4424bdf-56bf-41d4-a44e-cb11c1fc8c82)

### X_train_grad value
![ML55](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/2b484411-4120-43f6-952a-72926103cf13)

### Y_train_grad value
![ML56](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/cda1ea39-ff28-4271-a8b3-167cb421bb12)

### Print(res.x)
![ML57](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/41ab779b-8d86-42db-8ddc-500acf348a4c)

### Decision Boundary graph for Exam Score

![ML58](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/9b4fa63e-b92e-4f00-a671-695146d39672)

### Probability value

![ML59](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/584ef865-086f-4ada-84de-b7dc51813cfe)

### Prediction value of mean

![ML510](https://github.com/roshiniRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118956165/5e9b4600-c7d9-4153-b151-e9b47a1bc5ea)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

