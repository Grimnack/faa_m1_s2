#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP1 FAA Matthieu Caron 2016

import matplotlib.pyplot as plt
import numpy as np
import math
import random as r




################# LECTURE #################

def lecture(pathname) :
    fichier = open(pathname,'r')
    res = []
    for ligne in fichier :
        res.append(float(ligne))
    fichier.close()
    return res 

def lecture2(pathname) :
    fichier = open(pathname,'r')
    (dim1,dim2) = ([],[])
    for ligne in fichier :
        (taille,poids) = ligne.split()
        dim1.append(float(taille))
        dim2.append(float(poids))
    return (dim1,dim2) 

################# VARIABLES #################
x = np.linspace(4,15,100)

x0 = np.array(lecture('x0.txt'),float)
x1 = np.array(lecture('x1.txt'),float)
x2 = np.array(lecture('x2.txt'),float)

y0 = np.array(lecture('y0.txt'),float)
y1 = np.array(lecture('y1.txt'),float)
y2 = np.array(lecture('y2.txt'),float)

(tmp_taille_f,tmp_poids_f) = lecture2('taillepoids_f.txt')
(tmp_taille_h,tmp_poids_h) = lecture2('taillepoids_h.txt')
taille_f = np.array(tmp_taille_f,float)
taille_h = np.array(tmp_taille_h,float)

nbFilles = len(taille_f)
nbHommes = len(taille_h)

poids_f = np.array(tmp_poids_f,float)
poids_h = np.array(tmp_poids_h,float)

lesZeros = np.zeros(nbFilles)
lesUns = np.ones(len(taille_h))

ones0 = np.ones(len(x0))
ones1 = np.ones(len(x1))
ones2 = np.ones(len(x2))

matrix0 = np.zeros((2,len(x0)))
matrix1 = np.zeros((2,len(x1)))
matrix2 = np.zeros((2,len(x2)))

matrix0[1,:] = x0
matrix1[1,:] = x1
matrix2[1,:] = x2

matrix0[0,:] = ones0
matrix1[0,:] = ones1
matrix2[0,:] = ones2
############# FONCTIONS #############
def sigmoidMatrix(matrix) :
    for ligne in matrix :
        for x in ligne :
            x = sigmoid(x)

def sigmoid(x) :
    return 1 / (1+math.exp(-x))

def matrixDegresN(degres, vecteur) :
    '''
    entree degres (int) 
    sortie matrix de taille 
    '''
    newMatrix = np.zeros((degres+1,len(vecteur)))
    newMatrix[0,:] = np.ones(len(vecteur))
    for i in range(1,degres+1) :
        for j in range(len(vecteur)) :
            newMatrix[i][j] = vecteur[j]**i
    return newMatrix

def donneY(teta,matrix) :
    return np.dot(matrix.T,teta)
############# MESURE DE PERF #############

def mesureAbs(x1,y,teta,N=100) :
    vecteur = y - np.dot(x1.T,teta)
    return np.sum(np.absolute(vecteur))/N

def mesureNormal2(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.dot(vecteur.T, vecteur)/N

def mesureNormal1(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    interm = np.dot(vecteur.T, vecteur)
    return math.sqrt(interm)/N

def mesureNormalinf(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.amax(np.absolute(vecteur))


def moindresCarres(matrix, vecteur):
    gauche = np.dot(matrix, matrix.T)
    droite = np.dot(matrix,vecteur) 
    return  np.dot(np.linalg.inv(gauche),droite)

def alpha(t) :
    return 1./(1.+4000.*float(t))

def descenteGradient(x1,y,teta,t, epsilon=0.00000001,N=100) :
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        intermediaire = np.dot(x1,(y - np.dot(x1.T,tetaActuel))) #gradient
        tetaPlusPlus = tetaActuel + np.dot(np.dot(alpha(t),intermediaire),1./float(N))
        #test de convergence 
        if math.fabs(mesureNormal2(x1,y,tetaPlusPlus)-mesureNormal2(x1,y,tetaActuel)) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1

def descenteGradientStochastique(matrix,y,teta,t,epsilon=0.00000001,N=100) :
    '''
    a corriger
    '''
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        i = r.randint(0,99)
        vecX = np.array([matrix[0][i],matrix[1][i]],float)
        intermediaire = np.dot(vecX,(y[i] - np.dot(tetaActuel.T,vecX))) #gradient
        tetaPlusPlus = tetaActuel + np.dot(np.dot(alpha(t),intermediaire),1./float(N))
        #test de convergence 
        if math.fabs(mesureNormal2(vecX,y[i],tetaPlusPlus)-mesureNormal2(vecX,y[i],tetaActuel)) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1

def descenteGradientSigmoide(matrix,y,teta,t,epsilon,N):
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        intermediaire = np.dot(matrix,(y - sigmoid(np.dot(matrix.T,tetaActuel))))
        tetaPlusPlus = tetaActuel + np.dot(nd.dot(alpha(t),intermediaire),1./float(N))
        if math.fabs(mesureNormal2(matrix,y,tetaPlusPlus)-mesureNormal2(matrix,y,tetaActuel)) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1

def crossValidationSet(X,Y) :
    taille = len(Y)
    Xlearn = X[:(taille//10)*9]
    Ylearn = Y[:(taille//10)*9]
    Xtest = X[(taille//10)*9+1:]
    Ytest = Y[(taille//10)*9:]
    return (Xlearn,Ylearn,Xtest,Ytest)

def donneErreur(Ytest,yCompare) :
    print "tailles ", len(Ytest), len(yCompare)
    for i in range(len(yCompare)) :
        print Ytest[i] , yCompare[i]


################# SCRIPT #################

############## Matrix 0 ##############
# plt.plot(x0,y0,'r.',label="donnee")
# matrix = matrixDegresN(2,x0)
# teta = moindresCarres(matrix,y0)
# y = donneY(teta,matrix)
# plt.plot(x0,y,'b',label="approximation")

############## Matrix 1 ##############
# plt.plot(x1,y1,'r.',label="donnee")
# matrix = matrixDegresN(2,x1)
# teta = moindresCarres(matrix,y1)
# y = donneY(teta,matrix)
# plt.plot(x1,y,'b',label="approximation")

############## Matrix 2 ##############
# plt.plot(x2,y2,'r.',label="donnee")
# matrix = matrixDegresN(20,x2)
# teta = moindresCarres(matrix,y2)
# y = donneY(teta,matrix)
# plt.plot(x2,y,'b.',label="approximation")



# plt.legend()
# plt.show()

############## Cross Validation ##############

############## Matrix 0 ##############
(Xlearn,Ylearn,Xtest,Ytest) = crossValidationSet(x0,y0)
learn = matrixDegresN(2,Xlearn)
test = matrixDegresN(2,Xtest)
teta = moindresCarres(learn,Ylearn)
yCompare = donneY(teta,test)
donneErreur(Ytest,yCompare)

############## Matrix 1 ##############

############## Matrix 2 ##############



