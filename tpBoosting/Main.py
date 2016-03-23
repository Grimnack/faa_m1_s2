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
    Ytest = Y[(taille//10)*9+1:]
    return (Xlearn,Ylearn,Xtest,Ytest)

# (diabeteX,diabeteY,diabeteXtest,diabeteYtest) = lecture('pima-indians-diabetes.data')
(spamX,spamY,spamXtest,spamYtest) = lecture('spambase.data')


N = len(spamX)
spamW = []
Wi = 1/N
print Wi
for i in range(N) :
    spamW.append(Wi)  


classifier = tree.DecisionTreeClassifier(max_depth=1)

def adaboost(X,Y,W,nbClassifier) :
    '''
    ATTENTION MOTHER FUCKER 
    transformer les 0 de y en -1 pour faire marcher l'algo


    Diapo 6 slide boosting 
    http://www.lifl.fr/~pietquin/teaching/faa-m1s2-boosting.pdf
    '''
    N = len(X)
    Wt = np.copy(W)
    HT = []
    for t in range(nbClassifier) :
        # Entrainer un classifier
        classifier = tree.DecisionTreeClassifier(max_depth=prof)
        classifier = classifier.fit(X,Y,sample_weight=Wt)
        # Calculer le taux d erreur
        tab = classifier.predict(X)
        nbErreur = 0 
        for i in len(tab) :
            if tab[i] != Y[i] :
                nbErreur += Wt[i]*1
        epsilon = nbErreur / N
        # Calculer le pas d apprentissage alpha
        alpha = 0.5 * np.log((1 - epsilon)/epsilon)    # np.log is ln
        # Mettre a jour Wt 
        interm = []
        for i in range(N) :
            interm.append(Wt[i]*np.exp(-alpha*Y[i]*tab[i]))
        Wt[i] = interm[i]/sum(interm)
        HT.append((classifier,alpha))
    return HT








# for prof in range(1,20) :
#     classifier = tree.DecisionTreeClassifier(max_depth=prof)
#     classifier = classifier.fit(diabeteX,diabeteY)
#     nbError = 0
#     for i in range(len(diabeteXtest)) :
#         tab = classifier.predict(diabeteXtest[i])
#         for x in tab :
#             # print tab[0],diabeteYtest[i]
#             if tab[0] != diabeteYtest[i] :
#                 nbError = nbError + 1

#     tauxErr = float(nbError)/float(len(diabeteYtest))
#     print "pour une profondeur de ",prof," voici le taux d erreur ",tauxErr