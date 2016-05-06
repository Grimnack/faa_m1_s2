import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import datasets
from sklearn import svm
import numpy as np





######################## GRAPHIQUE 1 ######################## 
X, Y = datasets.make_blobs(centers=2,n_samples=500,cluster_std=0.3,random_state=0)
clf = svm.SVC(C=1.0, kernel='linear',shrinking=False, probability=False,max_iter=500)
######################## GRAPHIQUE 2 ########################
# X, Y = datasets.make_blobs(centers=2,n_samples=500,cluster_std=0.3,random_state=0)
# clf = svm.SVC(C=0.001, kernel='linear',shrinking=False, probability=False,max_iter=500)
######################## GRAPHIQUE 3 ########################
# X, Y = datasets.make_blobs(centers=2,n_samples=500,cluster_std=0.8,random_state=0)
# clf = svm.SVC(C=1.0, kernel='linear',shrinking=False, probability=False,max_iter=500)

plt.scatter(X[:,0],X[:,1],c=Y)
clf.fit(X,Y)

hY = clf.predict(X)
error = np.mean(abs(hY-Y))

b = clf.intercept_
# w = np.array(clf.coef_)
w = clf.coef_[0]
a = -w[0] / w[1]
x = np.linspace(-1,5)
fx = a*x - b[0]/w[1]
#ici la marge a ete trouve a l'arrache... peut etre en rapport avec clf et peut etre C
marge1 = fx + 1/w[1]
marge2 = fx - 1/w[1]

plt.plot(x,fx)
plt.plot(x,marge1)
plt.plot(x,marge2)
for point in clf.support_vectors_ :
    plt.plot(point[0],point[1],'cs')
plt.show()
