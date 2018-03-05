#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import random
import sys


def gradientDescent(x, y, th, eps, m, iteration):
	xTrans = x.transpose()
	c=1
	while c!=0 :
		for i in range(0, iteration):
			h = np.dot(x, th)
			loss = h - y
			c=0
			cost = np.sum(loss ** 2) / (2 * m)
			#print('Cost:',cost)
			gradient = np.dot(xTrans, loss) / m
			th = th - eps * gradient
			c=round(cost,2)
			#print c
		eps=eps+0.00001
	return th, eps

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)
var1=sys.argv[1] if len(sys.argv) > 1 else "0"
var2=sys.argv[2] if len(sys.argv) > 2 else "0"
var3=sys.argv[3] if len(sys.argv) > 3 else "0"
#print 'var1=',var1
#print 'var2=',var2
#print 'var3=',var3
x2=int(var1)
x1=int(var2)
x0=int(var3)
#var5=x2-x1
#var6=x0+x2-x1
#print 'var5=',var5
#print 'var6=',var6
i=0
x = np.zeros(shape=(20, 3))
y = np.zeros(shape=20)
#print x
#print y
for num in range(-10,10): 
    x[i][0]=1
    x[i][1]=num
    x[i][2]=num**2
    y[i]=(((x2)*num**2)+((x1)*num)+x0)
    i=i+1 
#print x
#print y
#print x[5]
#print y[12]
#print x
#print y
m, n = np.shape(x)
iteration = 5000
eps = 0.000001
th = np.ones(n)
noise = np.random.uniform(-0.1,0.1,y.shape)
y = y + noise
th, eps = gradientDescent(x, y, th, eps, m, iteration)
print 'learning rate=',eps
xx0=th[0]
xx1=th[1]
xx2=th[2]
print 'x2=',xx2
print 'x1=',xx1
print 'x0=',xx0
j=0
xx = np.zeros(shape=20)
yy = np.zeros(shape=20)
#print x
#print y
for num in range(-10,10): 
    xx[j]=num
    yy[j]=(((xx2)*num**2)+((xx1)*num)+xx0)
    j=j+1 
plt.plot(x,y,'kx',xx,yy,)
plt.show()
