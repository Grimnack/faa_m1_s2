#!/usr/bin/env python
#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np

plt.xlabel("erreur sur la base de test")
plt.ylabel("nombre d iterations")

############# Usefull functions #################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)




#################### Defining size of input and output ####################
# Define the size of the input and of the output
x = tf.placeholder(tf.float32, shape=[None, 784])
# Here None means that a dimension can be of any length
y_ = tf.placeholder(tf.float32, shape=[None, 10])



#################### Building the DNN #####################
########## Declaring weights of layer 1 and 2 ##########
# Layer 1
W1 = weight_variable([784,350])
b1 = bias_variable([350])

# Layer 2
W2 = weight_variable([350,200])
b2 = bias_variable([200]) 

# Layer 3
W3 = weight_variable([200,100])
b3 = bias_variable([100])

# Layer 4
W4 = weight_variable([100,50])
b4 = bias_variable([50])

# Layer 5
W5 = weight_variable([50,10])
b5 = bias_variable([10])


########## Building the intermediate error ##########

def config1() :
  W1 = weight_variable([784,10])
  b1 = bias_variable([10])
  a1 = tf.sigmoid(tf.matmul(x,W1) + b1)
  y_out = tf.nn.softmax(a1)
  return (a1,y_out)

def config2() :
  # Layer 1
  W1 = weight_variable([784,200])
  b1 = bias_variable([200])

  # Layer 2
  W2 = weight_variable([200,10])
  b2 = bias_variable([10])
  ##################################
  a1 = tf.sigmoid(tf.matmul(x,W1) + b1)
  a2 = tf.sigmoid(tf.matmul(a1,W2) + b2)
  y_out = tf.nn.softmax(a2)
  return (a1,a2,y_out)

def config5() :
  # Layer 1
  W1 = weight_variable([784,350])
  b1 = bias_variable([350])
  # Layer 2
  W2 = weight_variable([350,200])
  b2 = bias_variable([200]) 
  # Layer 3
  W3 = weight_variable([200,100])
  b3 = bias_variable([100])
  # Layer 4
  W4 = weight_variable([100,50])
  b4 = bias_variable([50])
  # Layer 5
  W5 = weight_variable([50,10])
  b5 = bias_variable([10])
  ##################################
  # Layer 1
  a1 = tf.matmul(x,W1) + b1
  # Layer 2
  a2 = tf.matmul(a1,W2) + b2
  # Layer 3
  a3 = tf.matmul(a2,W3) + b3
  # Layer 4
  a4 = tf.matmul(a3,W4) + b4
  # Layer 5
  a5 = tf.matmul(a4,W5) + b5
  # y_out
  y_out = tf.nn.softmax(a5)
  return (a1,a2,a3,a4,a5,y_out)

#################### Defining the loss #####################

########### ICI COMMENTER/DECOMMENTER EN FONCTION DE 1,2 ou 5 couches souhaitée
# (a1,y_out) = config1()
(a1,a2,y_out) = config2()
# (a1,a2,a3,a4,a5,y_out) = config5()
###############################################################################

# Loss cross entropy
cross_entropy = - tf.reduce_sum(y_ * tf.log(y_out))
# with a L2 regularization
regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1)
LAMBDA = 0.0001

loss = cross_entropy# + LAMBDA*regularizer



#################### Training method #####################
# Training with gradient descent
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)



#################### Creating a session and initializing variables #####################
# Creating a session
sess = tf.Session()
# Initialize variables
sess.run(tf.initialize_all_variables())



#################### Prediction error #####################
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Gradient descent with minibatch of size 100
list_err = []
list_err_test = []
list_iteration = []


#################### Optimization #####################
for i in range(20000):
    # Creating the mini-batch
    batch_xs, batch_ys = mnist.train.next_batch(20)
    # running one step of the optimization method on the mini-batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%100 == 0:
        # train error computation
        acuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        print "###################################################"
        print "step %d, training err %g"%(i, 1-acuracy)
        list_iteration.append(i)
        list_err.append(1-acuracy)
        acuracy_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        list_err_test.append(1- acuracy_test )

print "test accuracy %g"%(acuracy)
line1, = plt.plot(list_iteration,list_err,label="erreur")
line2, = plt.plot(list_iteration,list_err_test,label="erreur test")
plt.legend(handles=[line1, line2])
# plt.show()



#################### TP: Tester plusieurs architectures ####################
# Telecharger les 2 script.
# Creez des reseaux de neurones a 1,2 et 5 couches.
# Testez differentes configurations pour les couches intermediaires (RLU, tanh ou sigmoide).
# Pourquoi faut-il initialiser les couches RLU avec un biais positif?
# Optimiser avec une des methodes de votre choix (testez au moins AdaGrad, SGD et SGD avec un learning rate decay geometrique).
# Pour chacun des tests: Affichez l'erreur sur la base d'entrainement et sur la base de test en fonction des iterations et creez la matrice de confusion sur la base de test.