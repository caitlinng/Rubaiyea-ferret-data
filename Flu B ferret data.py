#!/usr/bin/env python
# coding: utf-8

# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''
# Creating arrays
days = range(1,12)

OST_n1 = np.array([days, [0.5,0.5,0.5,2.5,3.833333,3.5,3.166667,0.833333,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,0.5,3.5,2.5,3.5,3.166667,2.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,0.833333,1.166667,2.833333,4.166667,0.833333,1.166667,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.833333,2.833333,2.833333,3.166667,1.166667,0.833333,0.5]])

plc_n1 = np.array([days, [0.5,0.5,0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,0.5,0.5,4.166667,3.5,3.5,2.833333,0.5,0.5,0.5]])
plc_n3 = np.array([days, [0.5,0.5,0.5,3.833333,4.833333,3.5,3.5,0.5,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,0.5,0.5,1.166667,3.833333,3.5,2.833333,3.166667,0.5,0.5,0.5]])

OST_list = np.array([OST_n1, OST_n2, OST_n3, OST_n4])
plc_list = np.array([plc_n1, plc_n2, plc_n3, plc_n4])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2], [plc_n3, plc_n4]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('Influenza B OST-treated ferrets')

for i in range(2):
    for j in range(2):
        x = OST_all[i, j][0] # Days
        y = OST_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        axs[i, j].set(xlim=[1, 11],ylim=[0, 6]) 
        
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')
            

fig2, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig2.suptitle('Influenza B placebo-treated ferrets')

for i in range(2):
    for j in range(2):
        x = plc_all[i, j][0] # Days
        y = plc_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        axs[i, j].set(xlim=[1, 11],ylim=[0, 6]) 
        
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')

plt.show()
    


# In[66]:


# Finding AUC (no value adjustment)
''' 
Finding area under the curve (AUC)
numpy.trapz(y, x=None, dx=1.0, axis=-1)
'''
AUC_OST = []
n = 0
#OST_all = OST_all.flatten()

for i in OST_list:
    n += 1
    ferret = str(n)
    AUC = np.trapz(i[1], i[0])
    AUC_OST.append((ferret, AUC))

AUC_plc = []
n = 0
for i in plc_list:
    n += 1
    ferret = str(n)
    AUC = np.trapz(i[1], i[0])
    AUC_plc.append((ferret, AUC))
    
fig2, axs = plt.subplots(1,2)
fig2.suptitle('Area under the curve (all)')
OST_x = [] # Ferret name
OST_y = [] # AUC

for i in AUC_OST:   
    OST_x.append(i[0])
    OST_y.append(i[1])

axs[0].bar(OST_x, OST_y)

axs[0].set_title('OST-treated')
axs[0].set_xlabel('Ferret')
axs[0].set_ylabel('AUC')

# Placebo-treated ferrets plot
plc_x = [] # Ferret name
plc_y = [] # AUC

for i in AUC_plc:   
    plc_x.append(i[0])
    plc_y.append(i[1])

axs[1].bar(plc_x, plc_y)

axs[1].set_title('Placebo-treated')
axs[1].set_xlabel('Ferret')

axs[0].set_ylim(0, 30)
axs[1].set_ylim(0, 30)

plt.show()


# In[72]:


# Finding AUC (baseline removed)
''' 
Finding area under the curve (AUC)
numpy.trapz(y, x=None, dx=1.0, axis=-1)
'''
AUC_OST = []
n = 0
#OST_all = OST_all.flatten()

for i in OST_list:
    n += 1
    ferret = str(n)
    y_new = i[1] - 0.5
    AUC = np.trapz(y_new, i[0])
    AUC_OST.append((ferret, AUC))

AUC_plc = []
n = 0
for i in plc_list:
    n += 1
    ferret = str(n)
    y_new = i[1] - 0.5
    AUC = np.trapz(y_new, i[0])
    AUC_plc.append((ferret, AUC))
    
fig2, axs = plt.subplots(1,2)
fig2.suptitle('Area under the curve (above baseline)')
OST_x = [] # Ferret name
OST_y = [] # AUC

for i in AUC_OST:   
    OST_x.append(i[0])
    OST_y.append(i[1])

axs[0].bar(OST_x, OST_y)

axs[0].set_title('OST-treated')
axs[0].set_xlabel('Ferret')
axs[0].set_ylabel('AUC')

# Placebo-treated ferrets plot
plc_x = [] # Ferret name
plc_y = [] # AUC

for i in AUC_plc:   
    plc_x.append(i[0])
    plc_y.append(i[1])

axs[1].bar(plc_x, plc_y)

axs[1].set_title('Placebo-treated')
axs[1].set_xlabel('Ferret')

axs[0].set_ylim(0, 25)
axs[1].set_ylim(0, 25)

plt.show()


# In[ ]:




