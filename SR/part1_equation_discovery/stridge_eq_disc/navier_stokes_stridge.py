#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:45:04 2019

@author: sayin
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg 
#                                         projection='3d' in the call to fig.add_subplot
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

##################################################################################
##################################################################################
#
# Functions for sparse regression.
#               
##################################################################################
##################################################################################

def TrainSTRidge(R, Ut, lam, d_tol, maxit = 50, STR_iters = 50,
                 l0_penalty = None, normalize = 0, 
                 split = 0.8, print_best_tol = True):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a
    training set,then evaluates them  using a loss function on a holdout set.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
#    print d_tol
    tol = d_tol
    
    if l0_penalty == None:
        l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=-1)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):
#        print iter
        # Get a set of coefficients and error
        w = STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)
        test_mse =  mean_squared_error(TestY,  TestR.dot(w.real))

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal tolerance:", tol_best)

    return  w_best, tol_best, test_mse

def STRidge(X0, y, lam, maxit, tol, normalize, print_results = True):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    

    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d), X.T.dot(y),rcond=-1)[0]
    else: w = np.linalg.lstsq(X,y)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]   
#    print biginds.dtype
    
    # Threshold and continue
    for j in range(maxit):
#        print j
        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
#        print smallinds
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
#                if print_results: print "Tolerance too high -
#                                    all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
#        print biginds
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) +\
                      lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=-1)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=-1)[0]
    
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return  w  
    
#%%
def checksol(data): 
    u          = data.u.values
    ux         = data.ux.values
    u2x         = data.u2x.values
    ut         = data.ut.values  # this is our target, now mapped to Y
    
    predPE =  -u*ux + 0.01*u2x
    
    print("Mean squared error:", mean_squared_error(data.ut, predPE))
    print("R2 score :", r2_score(data.ut,predPE))
    
    plt.figure()
    x = np.arange(0,len(ut))
    plt.plot(x,predPE)       # predictions are in blue
    plt.plot(x,data.ut)       # actual values are in orange
    plt.show()    
    
#doublecheck the data is there
print(os.listdir("..\eit\SR\part1_eqation_discovery\data"))

# read in the data to pandas
if(platform.system() == 'Windows'): #Windows
    #doublecheck the data is there
    print(os.listdir("..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\part1_eqation_discovery\data"))
    # read in the data to pandas
    navier_data = pd.read_csv("..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\part1_eqation_discovery\data\\navier_stokes_data_u.csv",  encoding='utf-8')
else:
    print(os.listdir("/lustre1/home/gustavoo/TTK4853-EiT-Hybrid-modellering-og-analyse/SR/part1_eqation_discovery/data"))
    navier_data = pd.read_csv("/lustre1/home/gustavoo/TTK4853-EiT-Hybrid-modellering-og-analyse/SR/part1_eqation_discovery/data/navier_stokes_data_u.csv",  encoding='utf-8')


data = data.iloc[::10]
#checksol(data)


data1 = np.array(data)
theta = data1[:,:-1]
px = data1[:,-1:]

#x = np.linspace(0, 29, 30)
#y = np.linspace(0, 29, 30)
#X, Y = np.meshgrid(x, y)
#Z = np.zeros((30,30))
#for j in range(11):
#    for i in range(30):
#        Z[i] = theta[i*30:(i+1)*30,j]
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    ax.contour3D(X, Y, Z, 50, cmap='binary')
#    plt.show()


#plt.plot(theta[0:29,0])
#plt.show()
#plt.plot(theta[30:59,0])
#plt.show()
#plt.plot(px[0:29])
#plt.show()

c = np.ones((theta.shape[0],1))
theta_ = np.column_stack((c,theta))  ## Stack const =1--columnn  to theta
theta = np.delete(theta_, 2, axis=1)

lam1 = data.columns.get_loc("uux")  + 1   #uux
lam2 = data.columns.get_loc("vuy")  + 1   #uvy
lam3 = data.columns.get_loc("wuz")  + 1   #uwz
lam4 = data.columns.get_loc("u2x")  + 1   #u2x
lam5 = data.columns.get_loc("u2y")  + 1   #v2y
lam6 = data.columns.get_loc("u2z")  + 1   #w2z
#%%
key= '''
#Using STRidge to predict Navier Stokes
'''

n_alphas = 100
alphas = np.logspace(-10, -0.1 , n_alphas)
dtol = 0.01
otol = np.empty(n_alphas)
coefs = []
test_mse = np.empty(n_alphas)

coefs = []
for i, a in enumerate(alphas):
    w, otol[i], test_mse[i] = TrainSTRidge(theta,px,a,dtol,20)
    coefs.append(w) 

coef = np.array(coefs)
cf = coef.reshape(coef.shape[0],coef.shape[1])
#%%
coef = np.array(coefs)

ax = plt.gca()
for i in range(coef.shape[1]):
    if i ==lam1:
        ax.plot(alphas, coef[:,i],'b--',lw=2, label = '$uu_{x}$')
    elif i ==lam2:
        ax.plot(alphas, coef[:,i]/0.01,'r--',lw=2, label = '$vu_{y}$')
        pass
    elif i ==lam3:
        ax.plot(alphas, coef[:,i]/0.01,'g--',lw=2, label = '$wu_{z}$')
        pass
    elif i ==lam4:
        ax.plot(alphas, coef[:,i]/0.01,'m--',lw=2, label = '$u_{2x}$')
        pass
    elif i ==lam5:
        ax.plot(alphas, coef[:,i]/0.01,'k--',lw=2, label = '$u_{2y}$')
        pass
    elif i ==lam6:
        ax.plot(alphas, coef[:,i]/0.01,'w--',lw=2, label = '$u_{2z}$')
        pass
    else:
        ax.plot(alphas, coef[:,i], lw=2)    
        pass 
      
ax.set_xscale('log')
ax.set_xlim(np.max(alphas),np.min(alphas))  # reverse axis
plt.xlabel('alpha', size = 15,labelpad=0.2)
plt.ylabel('Coefficients', size = 15,labelpad=0.2)
plt.show()

a = np.vstack([[data.keys().tolist()[0:len(data.keys().tolist())-1]],coefs[-1].T])
print("A:")
print(a)
np.savetxt('navier_coef.txt',a,fmt='%s')#, coef=cf, dtol= otol, lam=alphas, test_mse=test_mse)