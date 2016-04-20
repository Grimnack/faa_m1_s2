import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import datasets
from sklearn import svm
import numpy as np

X, Y = datasets.make_blobs(centers=2,n_samples=500,cluster_std=0.3,random_state=0)

plt.scatter(X[:,0],X[:,1],c=Y)

clf = svm.SVC(C=1.0, kernel='linear',shrinking=False, probability=False,max_iter=500)
clf.fit(X,Y)

hY = clf.predict(X)
error = np.mean(abs(hY-Y))

b = clf.intercept_
# w = np.array(clf.coef_)
w = clf.coef_[0]
a = -w[0] / w[1]
x = np.linspace(-1,5)
fx = a*x - b[0]/w[1]

plt.plot(x,fx)
plt.show()
