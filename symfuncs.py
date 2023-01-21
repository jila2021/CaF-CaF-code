# -*- coding: utf-8 -*-
"""
Created on July-5 2022

@author: Dibyendu Sadar
"""

import numpy as np

# Join functions f1 and f2 with a sigmoid function as a function of x in range given by mergrangre
def joinfunctions2(x,coords,f1,f2,mergrange):
      merg1=float(mergrange[0])
      merg2=float(mergrange[1])
      F=np.zeros(len(x))
      ind1=np.where(x<=merg1)[0]
      if len(ind1)>0:
            F[ind1]=f1(coords[ind1,:])
      ind2=np.where(x>=merg2)[0]
      if len(ind2)>0:
            F[ind2]=f2(coords[ind2,:])
      ind3=np.where((x>merg1)&(x<merg2))[0]
      if len(ind3)>0:
            mergmid=(merg1+merg2)/2.
            mergwidth=(merg2-merg1)/2.
            y2=0.5+9./16.*np.sin(np.pi*(x[ind3]-mergmid)/2./mergwidth)+1./16.*np.sin(3.*np.pi*(x[ind3]-mergmid)/2./mergwidth)
            y1=1-y2
            F[ind3]=f1(coords[ind3,:])*y1+f2(coords[ind3,:])*y2
      return F

#Permutation which exchanges coordinates corresponding to an exchange of both atoms A and B
def perm1(coAD):
	cor=np.zeros(np.shape(coAD))
	cor[:,0]=np.copy(coAD[:,0])
	cor[:,1]=np.copy(coAD[:,4])
	cor[:,4]=np.copy(coAD[:,1])
	cor[:,2]=np.copy(coAD[:,3])
	cor[:,3]=np.copy(coAD[:,2])
	cor[:,5]=np.copy(coAD[:,5])
	return cor

#Permutation which exchanges coordinates corresponding to an exchange of both atoms F
def perm2(coAD):
	cor=np.zeros(np.shape(coAD))
	cor[:,0]=np.copy(coAD[:,0])
	cor[:,1]=np.copy(coAD[:,3])
	cor[:,3]=np.copy(coAD[:,1])
	cor[:,2]=np.copy(coAD[:,4])
	cor[:,4]=np.copy(coAD[:,2])
	cor[:,5]=np.copy(coAD[:,5])
	return cor

#Permutation which exchanges coordinates corresponding to an exchange of both atoms Ca
def perm3(coAD):
	cor=np.zeros(np.shape(coAD))
	cor[:,0]=np.copy(coAD[:,0])
	cor[:,1]=np.copy(coAD[:,2])
	cor[:,2]=np.copy(coAD[:,1])
	cor[:,3]=np.copy(coAD[:,4])
	cor[:,4]=np.copy(coAD[:,3])
	cor[:,5]=np.copy(coAD[:,5])
	return cor

#Symmetrize GP in case only the AB-AB arrangement is incorporated
#One needs to decide for which permutation the test points are best located compared to the training point region
# Then the results from the permutations are smoothly joined together
def symGP(GP):
    #Define permuted functions
    F1=lambda coAD: GP.predict(1/coAD) #no permutation
    F2=lambda coAD: GP.predict(1/perm1(coAD)) #Ca & F both exchange
    F3=lambda coAD: GP.predict(1/perm2(coAD)) #F exchanges
    F4=lambda coAD: GP.predict(1/perm3(coAD)) #Ca exchanges
    F_d1=lambda coAD: joinfunctions2(coAD[:,1]/(coAD[:,1]+coAD[:,4]),coAD,F1,F2,[0.4375,0.5625])
    F_d2= lambda coAD: joinfunctions2(coAD[:,1]/(coAD[:,1]+coAD[:,3]),coAD,F1,F3,[0.4375,0.5625])
    F_d3= lambda coAD: joinfunctions2(coAD[:,2]/(coAD[:,2]+coAD[:,3]),coAD,F4,F3,[0.4375,0.5625])
    F_d4= lambda coAD: joinfunctions2(coAD[:,2]/(coAD[:,2]+coAD[:,4]),coAD,F4,F2,[0.4375,0.5625])
    F_d12= lambda coAD: joinfunctions2(coAD[:,4]/(coAD[:,4]+coAD[:,3]),coAD,F_d1,F_d2,[0.4375,0.5625])
    F_d34= lambda coAD: joinfunctions2(coAD[:,3]/(coAD[:,3]+coAD[:,4]),coAD,F_d3,F_d4,[0.4375,0.5625])
    Ftot=lambda coAD: joinfunctions2(coAD[:,1]/(coAD[:,1]+coAD[:,2]),coAD,F_d12,F_d34,[0.4375,0.5625])
    return(Ftot)
