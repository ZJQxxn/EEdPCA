%pylab inline
from numpy import *
from numpy.random import rand, randn, randint
from dPCA import dPCA
# sys.path.append('../ee_dpca.py');
import ee_dpca
import sklearn.preprocessing
import matplotlib.pyplot as plt

# number of neurons, time-points and stimuli
N,T,S = 100,250,6

# noise-level and number of trials in each condition
noise, n_samples = 0.2, 10

# build two latent factors
zt = (arange(T)/float(T))
zs = (arange(S)/float(S))

# build trial-by trial data
trialR = noise*randn(n_samples,N,S,T)
trialR += randn(N)[None,:,None,None]*zt[None,None,None,:]
trialR += randn(N)[None,:,None,None]*zs[None,None,:,None]

# trial-average data
R = mean(trialR,0)

# center data
R -= mean(R.reshape((N,-1)),1)[:,None,None]

Ee_dpca=ee_dpca.EE_dPCA(labels='st',rho=8e-3, hyper_lamb=1e-4, hyper_tau=1e-2, regularizer=None)
Ee_dpca.protect = ['t']

ZZ = Ee_dpca.fit_transform(R,trialR)

time = arange(T)

figure(figsize=(16,7))
subplot(131)

for s in range(S):
    plot(time,ZZ['t'][0,s])

title('1st time component')
    
subplot(132)

for s in range(S):
    plot(time,ZZ['s'][0,s])
    
title('1st stimulus component')
    
subplot(133)

for s in range(S):
    plot(time,ZZ['st'][0,s])
    
title('1st mixing component')
show()

print(Ee_dpca.total_time)

dpca = dPCA.dPCA(labels='st',regularizer=None)
dpca.protect = ['t']

Z = dpca.fit_transform(R,trialR)

print(dpca.total_time)
print(Z['t'][1].shape)

time = arange(T)

figure(figsize=(16,7))
subplot(131)

for s in range(S):
    plot(time,Z['t'][0,s])
    print(Z['t'][0,s].shape)

title('1st time component')
    
subplot(132)

for s in range(S):
    plot(time,Z['s'][0,s])
    
title('1st stimulus component')
    
subplot(133)

for s in range(S):
    plot(time,Z['st'][0,s])
    
title('1st mixing component')
show()

import numpy as np
trail_data=np.load('./ElectricityUsage.npy',encoding = "latin1")

n_samples = 3
data = mean(trail_data,0)
N = 12
print(data.shape)
print(trail_data.shape)
data -= mean(data.reshape((N,-1)),-1)[:,None,None,None]
# print(data)

data = data.transpose(3,1,2,0)
trail_data = trail_data.transpose(0,4,2,3,1)
print(data.shape)


scaler = sklearn.preprocessing.MinMaxScaler()
for i in range(12):
    for j in range(20):
        data[i][j] = scaler.fit_transform(data[i][j])

dpca_ex=dPCA.dPCA(labels='mdic',regularizer=None)
Z_ex=dpca_ex.fit_transform(data,trail_data)


exv_ex = dpca_ex.explained_variance_ratio_
m=sum(exv_ex['m'])
d=sum(exv_ex['d'])
i=sum(exv_ex['i'])
c=sum(exv_ex['c'])
else_ele=sum(exv_ex['md']+exv_ex['mi']+exv_ex['mc']+exv_ex['di']+exv_ex['dc']+exv_ex['ic']+exv_ex['mdi']+exv_ex['mdc']+exv_ex['mic']+exv_ex['dic']+exv_ex['mdic'])
data_list = [m,d,i,c,else_ele]
print(data_list)
labels=['month','day','time interval','client','else']
# colors=['#9999ff','#ff9999','#7777aa','#2442aa','#dd5555']
colors=['yellowgreen','lightskyblue','peachpuff','purple','pink']
plt.axes(aspect='equal')
plt.pie(x = data_list, 
        labels=labels, 
        colors=colors,
        autopct='%.2f%%',
        startangle = 180,
        textprops = {'fontsize':12, 'color':'k'},
        center = (1.5,1.5),
        radius = 1.4)

explained_var_dpca=sum(data_list)
print(data_list)
print(explained_var_dpca)

ee_dpca_ex=ee_dpca.EE_dPCA(labels='mdic',rho=8e-3, hyper_lamb=1e-4,hyper_tau=1e-3, regularizer=None)
ZZ_ex=ee_dpca_ex.fit_transform(data,trail_data)

ee_exv_ex = ee_dpca_ex.explained_variance_ratio_
ee_m=sum(ee_exv_ex['m'])
ee_d=sum(ee_exv_ex['d'])
ee_i=sum(ee_exv_ex['i'])
ee_c=sum(ee_exv_ex['c'])
ee_else=sum(ee_exv_ex['md']+ee_exv_ex['mi']+ee_exv_ex['mc']+ee_exv_ex['di']+ee_exv_ex['dc']+ee_exv_ex['ic']+ee_exv_ex['mdi']+ee_exv_ex['mdc']+ee_exv_ex['mic']+ee_exv_ex['dic']+ee_exv_ex['mdic'])
ee_data_list = [ee_m,ee_d,ee_i,ee_c,ee_else]
print(ee_data_list)
labels=['month','day','time interval','client','else']
# colors=['#9999ff','#ff9999','#7777aa','#2442aa','#dd5555']
colors = ['yellowgreen','lightskyblue','peachpuff','purple','pink']

plt.axes(aspect='equal')
plt.pie(x = ee_data_list, 
        labels=labels, 
        colors=colors,
        autopct='%.2f%%',
        startangle = 180,
        radius = 1.4,
        textprops = {'fontsize':12, 'color':'k'},
        center = (1.5,1.5) )

ee_explained_var_dpca=sum(ee_data_list)
print(ee_data_list)
print(ee_explained_var_dpca)