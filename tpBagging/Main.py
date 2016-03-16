from sklearn import tree
import numpy as np

##### DONNEES ##### 

def lecture(pathname) :
    '''
    renvoie le couple (matrix, vecteur) 
    '''
    dataset = np.loadtxt(pathname, delimiter=",")
    X = dataset[:,0:-1]
    Y = dataset[:,-1]
    taille = len(Y)
    Xlearn = X[:(taille//10)*9]
    Ylearn = Y[:(taille//10)*9]
    Xtest = X[(taille//10)*9+1:]
    Ytest = Y[(taille//10)*9:]
    return (Xlearn,Ylearn,Xtest,Ytest)

(diabeteX,diabeteY,diabeteXtest,diabeteYtest) = lecture('pima-indians-diabetes.data')
(spamX,spamY,spamXtest,spamYtest) = lecture('spambase.data')


for prof in range(1,20) :
    classifier = tree.DecisionTreeClassifier(max_depth=prof)
    classifier = classifier.fit(diabeteX,diabeteY)
    nbError = 0
    for i in range(len(diabeteXtest)) :
        tab = classifier.predict(diabeteXtest[i])
        for x in tab :
            # print tab[i],diabeteYtest[i]
            if tab[i] != diabeteYtest[i] :
                nbError = nbError + 1

    tauxErr = float(nbError)/float(len(diabeteYtest))
    print "pour une profondeur de ",prof," voici le taux d erreur ",tauxErr
