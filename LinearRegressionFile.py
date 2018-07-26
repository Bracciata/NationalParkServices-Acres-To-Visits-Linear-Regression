from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os.path

def cleanAcres(x):
    x =  x[0:x.find('acres')-1]
    x=  removeCommas(x)
    x=divide(x)
    return float(x)
def cleanVisitors(x):
    x=removeCommas(x)
    x=divide(x)
    return float(x)
def removeCommas(y):
    y=y.replace(',','')
    return float(y)
def divide(z):
    z=z/1000000
    return float(z)

rng = numpy.random
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
 
# Training Data
dfWholeDoc = pd.read_excel(os.path.abspath(os.path.join(__file__, os.pardir))+"/NPSData.xlsx",header=None) 
dfAcres=dfWholeDoc.loc[2:61,4]
dfAcres=dfAcres.apply(cleanAcres)
train_X=numpy.asarray(dfAcres)

dfVisitors=dfWholeDoc.loc[2:61,5]
dfVisitors=dfVisitors.apply(cleanVisitors)
train_Y=numpy.asarray(dfVisitors)

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
