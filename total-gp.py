import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, WhiteKernel, ConstantKernel as C
from symfuncs import *
#from coordtransform import *
from sklearn.model_selection import train_test_split
from numpy import asarray
from numpy import savetxt
import pandas as pd
import csv
import pickle

cm1=1.0/2.19474631e5
data_array=np.loadtxt('GP6_invad.txt')
easymp=-273.09286

Emrciq = data_array[:,6]
inddel=np.where(Emrciq>np.amin(Emrciq)+0.2)[0]
data_array=np.delete(data_array,inddel,axis=0)

#If true then extra symmetric equivalent training points are added to the training set
add_sympoints=True
#If true, the fit is symmetrized when evaluating
symmetrize=True

#For R<Rsym, symmetrically equivalent points are added to the training set
Rsym=8

R = 1/data_array[:,0]  
r13 = data_array[:,1]
r23 = data_array[:,2]
r14 = data_array[:,3]
r24 = data_array[:,4]
r34 = data_array[:,5]
V = data_array[:,6]
V = V-easymp

        #coordinates transformation

#x_data_n = np.asarray([R,r13,r23,r14,r24,r34]).T    #.reshape(-1,1) #training data set

x_data=data_array[:,:6]#1/(x_data_n)
y_data = V

if add_sympoints==True:
    print('Number of points before adding extra symmetric points:', len(y_data))
    extrapoints=np.where(R<Rsym)[0]
    extrax=perm3(x_data[extrapoints,:])
    extray=y_data[extrapoints]
#    temp=np.copy(extrax[:,1])
#    extrax[:,1]=np.copy(extrax[:,2])
#    extrax[:,2]=np.copy(temp)
#    temp=np.copy(extrax[:,3])
#    extrax[:,3]=np.copy(extrax[:,4])
#    extrax[:,4]=np.copy(temp)	
    x_data=np.concatenate([x_data,extrax])
    y_data=np.concatenate([y_data,extray])
    print('Number of points after adding extra symmetric points:',len(y_data))

#print(x_data.shape)
print(y_data.shape)


#Define kernel
kernel =C(1.0, (1e-5, 1e5)) * Matern(length_scale=[1.0,1.0,1.0,1.0,1.0,1.0], length_scale_bounds=(1e-5, 1e5), nu=2.5) + WhiteKernel(noise_level=0.1)
#C(1.0, (1e-5, 1e5)) * 
#Construct GP fit
gp = GaussianProcessRegressor(kernel=kernel)
model = gp.fit(x_data, y_data)
#Display trained kernel and marginal Log-likelihood
print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f", gp.log_marginal_likelihood(gp.kernel_.theta))


#test data load
test_array=np.loadtxt('test_data.txt')
Emrciqt = test_array[:,6]
inddel=np.where(Emrciqt>np.amin(Emrciqt)+0.2)[0]
test_array=np.delete(test_array,inddel,axis=0)


r12t = test_array[:,0]  #.reshape(-1,1)
r13t = test_array[:,1]
r23t = test_array[:,2]
r14t = test_array[:,3]
r24t = test_array[:,4]
r34t = test_array[:,5]
Vt = test_array[:,6]
Vt = Vt-easymp

x_test_sp = np.asarray([r12t,r13t,r23t,r14t,r24t,r34t]).T  
y_test = Vt

x_test=1/x_test_sp 
Ntest=len(y_test)

if symmetrize==False:
    y_pred = gp.predict(1/x_test_sp)
if symmetrize==True:
    gpsym=symGP(gp)
    y_pred = gpsym(1/x_test_sp)

y_diff = y_test-y_pred
rms = np.sqrt(sum(y_diff*y_diff)/Ntest)/cm1
print('RMS test error', rms)


