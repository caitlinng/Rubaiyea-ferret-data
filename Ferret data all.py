#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''
# Creating arrays
days = range(1,12)
OST_n1 = np.array([days, [0.5, 4.83, 5.17, 3.5, 3.83, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
OST_n2 = np.array([days, [0.5, 0.5, 3.83, 3.83, 4.17, 3.17, 3.83, 0.5, 0.5, 0.5, 0.5]])
OST_n3 = np.array([days, [0.5, 0.5, 0.5, 0.5, 2.5, 4.17, 3.5, 3, 0.5, 0.5, 0.5]])
OST_n4 = np.array([days, [0.5, 0.5, 0.5, 0.5, 3, 4.83, 2.83, 3.17, 4.5, 3.83, 2.83]])

plc_n1 = np.array([days, [0.5, 3, 4.17, 4.5, 3.5, 3.83, 0.5, 0.5, 0.5, 0.5, 0.5]])
plc_n2 = np.array([days, [0.5, 0.5, 3.83, 3.83, 2.5, 2.83, 2.83, 0.5, 0.5, 0.5, 0.5]])
plc_n3 = np.array([days, [0.5, 0.5, 0.5, 0.5, 0.5, 4.83, 4.5, 3.83, 4.17, 0.5, 0.5]])
plc_n4 = np.array([days, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.17, 3.5, 3.5, 3.17, 3.17]])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2], [plc_n3, plc_n4]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('H1N1(H275Y) OST-treated ferrets')

for i in range(2):
    for j in range(2):
        x = OST_all[i, j][0] # Days
        y = OST_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')
            


fig2, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig2.suptitle('H1N1(H275Y) placebo-treated ferrets')

for i in range(2):
    for j in range(2):
        x = plc_all[i, j][0] # Days
        y = plc_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')

plt.show()


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''

# Creating arrays
days = range(1,12)
OST_n1 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])

plc_n1 = np.array([days, [0.5,0.5,4.17,4.5,2.83,3.5,2.5,0.5,0.5,0.5,0.5]])
plc_n2 = np.array([days, [0.5,3.83,4.5,4.17,3.17,4.5,2.5,3.5,3.5,2.83,4.5]])
plc_n3 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,0.5,0.5,0.5,2.5,3.17,3.5,4.17,3.17,0.5,0.5]])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2], [plc_n3, plc_n4]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('H1N1 OST-treated ferrets')

for i in range(2):
    for j in range(2):
        x = OST_all[i, j][0] # Days
        y = OST_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        
        axs[i, j].set(xlim=[1, 11],ylim=[0, 5]) 
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')
            

fig2, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig2.suptitle('H1N1 placebo-treated ferrets')

for i in range(2):
    for j in range(2):
        x = plc_all[i, j][0] # Days
        y = plc_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        axs[i, j].set(xlim=[1, 11],ylim=[0, 5]) 
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')

plt.show()


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
    


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''
# Creating arrays
days = range(1,12)
plc_n1 = np.array([days, [0.5,2.5,4.5,0.5,0.5,3.17,3.17,0.5,0.5,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,3.83,3.83,2.5,2.5,3.5,0.5,0.5,0.5,0.5]])
plc_n3 = np.array([days, [0.5,3.5,4.5,0.5,0.5,4.17,0.5,0.5,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,3.5,3.83,0.5,3,4.17,0.5,0.5,0.5,0.5,0.5]])
plc_n5 = np.array([days, [0.5,4.83,3.5,0.5,3.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n6 = np.array([days, [0.5,2.5,3.5,3.83,2.83,3.83,0.5,0.5,0.5,0.5,0.5]])
plc_n7 = np.array([days, [0.5,3.83,4.83,0.5,2.83,2.5,0.5,0.5,0.5,0.5,0.5]])
filler = np.array([days, np.zeros(11)])

OST_n1 = np.array([days, [0.5,0.5,0.5,2.83,3.83,2.83,0.5,0.5,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,4.17,2.83,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,2.5,2.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,2.5,0.5,0.5,0.5]])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2, plc_n3, plc_n4], [plc_n5, plc_n6, plc_n7, filler]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('H3N2 OST-treated ferrets')

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
            
fig2, axs = plt.subplots(2, 4, sharex='all', sharey='all')
fig2.suptitle('H3N2 placebo-treated ferrets')

for i in range(2):
    for j in range(4):
        x = plc_all[i, j][0] # Days
        y = plc_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        axs[i, j].set(xlim=[1, 11],ylim=[0, 6]) 
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')

plt.show()


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''
# Creating arrays
days = range(1,12)
plc_n1 = np.array([days, [0.5,2.5,4.5,0.5,0.5,3.17,3.17,0.5,0.5,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,3.83,3.83,2.5,2.5,3.5,0.5,0.5,0.5,0.5]])
plc_n3 = np.array([days, [0.5,3.5,4.5,0.5,0.5,4.17,0.5,0.5,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,3.5,3.83,0.5,3,4.17,0.5,0.5,0.5,0.5,0.5]])
plc_n5 = np.array([days, [0.5,4.83,3.5,0.5,3.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n6 = np.array([days, [0.5,2.5,3.5,3.83,2.83,3.83,0.5,0.5,0.5,0.5,0.5]])
plc_n7 = np.array([days, [0.5,3.83,4.83,0.5,2.83,2.5,0.5,0.5,0.5,0.5,0.5]])
filler = np.array([days, np.zeros(11)])

OST_n1 = np.array([days, [0.5,0.5,0.5,2.83,3.83,2.83,0.5,0.5,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,4.17,2.83,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,2.5,2.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,2.5,0.5,0.5,0.5]])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2, plc_n3, plc_n4], [plc_n5, plc_n6, plc_n7, filler]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('H3N2 OST-treated ferrets')

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
            


fig2, axs = plt.subplots(2, 4, sharex='all', sharey='all')
fig2.suptitle('H3N2 placebo-treated ferrets')

for i in range(2):
    for j in range(4):
        x = plc_all[i, j][0] # Days
        y = plc_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        axs[i, j].set(xlim=[1, 11],ylim=[0, 6]) 
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')

plt.show()


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''
# Creating arrays
days = range(1,12)
plc_n1 = np.array([days, [0.5,3.5,4.17,3.83,3.5,2.5,0.5,0.5,0.5,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,4.5,4.83,3.17,3.17,3.17,0.5,0.5,0.5,0.5]])
plc_n3 = np.array([days, [0.5,5.83,5.83,4.5,2.5,3.5,0.5,0.5,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,4.5,4.5,3.5,2.83,0.5,0.5]])

OST_n1 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,3.17,0.5,3.5,3.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,4.17,4.5,4.83,4.17,3.5,3,2.83,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2], [plc_n3, plc_n4]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('H1N1pdm09 OST-treated ferrets')

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
fig2.suptitle('H1N1pdm09 placebo-treated ferrets')

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


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''

# Creating arrays
days = range(1,12)
plc_n1 = np.array([days, [0.5,4.5,5.17,2.83,3.17,3.5,0.5,0.5,0.5,0.5,0.5]])
plc_n2 = np.array([days, [3.83,5.17,5.83,3.83,2.83,3.17,0.5,0.5,0.5,0.5,0.5]])
plc_n3 = np.array([days, [4.5,3.83,5.17,4.17,3.5,3.83,0.5,0.5,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,2.83,5.83,4.5,3.83,3.5,3.17,3.5,4.17,2.5,0.5]])

OST_n1 = np.array([days, [0.5,4.5,4.83,4.83,4.17,3.17,3.5,2.5,2.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,3.83,5.5,4.17,4.17,3,0.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,5.83,5.17,4.5,2.83,3.17,0.5,0.5,0.5,0.5,0.5]])
OST_n4 = np.array([days, [4,4.17,4.5,3.83,4.17,2.83,0.5,0.5,0.5,0.5,0.5]])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2], [plc_n3, plc_n4]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('H1N1pdm09(H275Y) OST-treated ferrets')

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
fig2.suptitle('H1N1pdm09(H275Y) placebo-treated ferrets')

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


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''

# Creating arrays
days = range(1,12)

plc_n1 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,2.5,4.166667]])
plc_n3 = np.array([days, [0.5,0.5,0.5,0.5,3.5,4.166667,4.833333,3,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,2.5,6.166667,3.166667,3.833333,0.5,3.5]])
plc_n5 = np.array([days, [0.5,4.83,3.5,0.5,3.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n6 = np.array([days, [0.5,0.5,0.5,0.5,0.5,4.166667,3.833333,3.833333,0.5,0.5,0.5]])
plc_n7 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n8 = np.array([days, [0.5,0.5,0.5,0.5,0.5,3.5,4.166667,0.5,3.166667,2.833333,0.5]])

OST_n1 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,0.5,3.5,4.166667,4.5,4.5,0.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,3.166667,4.5,3.5,3.833333]])
OST_n5 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,3.5,4.5,4.5,0.5,0.5]])
OST_n6 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,5.166667]])
OST_n7 = np.array([days, [0.5,0.5,0.5,0.5,0.5,3.166667,3.5,4.166667,5.166667,0.5,0.5]])
OST_n8 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])

OST_all = np.array([[OST_n1, OST_n2, OST_n3, OST_n4], [OST_n5, OST_n6, OST_n7, OST_n8]])
plc_all = np.array([[plc_n1, plc_n2, plc_n3, plc_n4], [plc_n5, plc_n6, plc_n7, plc_n8]])

fig1, axs = plt.subplots(2, 4, sharex='all', sharey='all')
fig1.suptitle('H3N2(E119V) OST-treated ferrets')

for i in range(2):
    for j in range(4):
        x = OST_all[i, j][0] # Days
        y = OST_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        
        axs[i, j].set(xlim=[1, 11],ylim=[0, 6]) 
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')
            

fig2, axs = plt.subplots(2, 4, sharex='all', sharey='all')
fig2.suptitle('H3N2(E119V) placebo-treated ferrets')

for i in range(2):
    for j in range(4):
        x = plc_all[i, j][0] # Days
        y = plc_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        axs[i, j].set(xlim=[1, 11],ylim=[0, 6]) 
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')

plt.show()


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''

# Creating arrays
days = range(1,12)

plc_n1 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])
plc_n3 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.833333,2.833333,4.166667,1.166667,1.166667,0.833333]])
plc_n4 = np.array([days, [0.5,0.5,0.5,2.833333,3.5,3.166667,2.833333,2.5,0.833333,0.5,0.5]])
plc_n5 = np.array([days, [0.5,0.5,2.833333,3.833333,4.5,4.166667,2.5,0.5,0.5,0.5,0.5]])
plc_n6 = np.array([days, [0.5,0.5,4.833333,2.833333,3.833333,3.5,0.833333,0.5,0.5,0.5,0.5]])
plc_n7 = np.array([days, [0.5,0.5,2.83,2.5,3.83,3.5,3.17,0.5,0.5,0.5,0.5]])
plc_n8 = np.array([days, [0.5,0.5,2.83,3.5,4.83,3.83,0.83,0.5,0.5,0.5,0.5]])

OST_n1 = np.array([days, [0.5,0.5,0.5,1.166667,4.166667,1.5,3.833333,1.166667,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,0.5,1.5,4.5,4.5,2.5,2.833333,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,0.5,3.166667,4.166667,2.5,0.5,0.5,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,5.166667,3.5,2.5,0.833333,0.5,0.5,0.5]])


OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2, plc_n3, plc_n4], [plc_n5, plc_n6, plc_n7, plc_n8]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('B(D197N) OST-treated ferrets')

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


fig2, axs = plt.subplots(2, 4, sharex='all', sharey='all')
fig2.suptitle('B(D197N) placebo-treated ferrets')

for i in range(2):
    for j in range(4):
        x = plc_all[i, j][0] # Days
        y = plc_all[i, j][1] # TCID50
        axs[i, j].plot(x, y)
        
        axs[i, j].set(xlim=[1, 11],ylim=[0, 6]) 
        if j == 0:
            axs[i, j].set_ylabel('Viral titre log(TCID50)')
            
        if i == 1:
            axs[i, j].set_xlabel('Days')

plt.show()


# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Units = Log10 TCID50/ml
OST_n1 = viral load of ferret 1 treated with OST over 11 days
plc_n1 = viral load of ferret 1 treated with placebo over 11 days

'''

# Creating arrays
days = range(1,12)

plc_n1 = np.array([days, [0.5,0.5,0.5,0.5,0.5,3.833333,1.5,2.833333,0.5,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.833333,5.166667,4.5,2.833333,0.833333,0.5]])
plc_n3 = np.array([days, [0.5,0.5,0.5,0.5,0.5,0.5,0.833333,3.5,3.5,2.833333,1.166667]])
plc_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.5,3.166667,4.5,3.5,0.5,0.5,0.833333]])

OST_n1 = np.array([days, [0.5,0.5,0.5,2.833333,3.5,2.5,2.833333,2.5,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,2.833333,4.5,3.166667,3.166667,0.833333,0.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,3.166667,3.166667,2.5,2.833333,0.833333,0.5,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,1.166667,2.833333,2.833333,4.166667,0.5,1.166667,0.5]])


OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2], [plc_n3, plc_n4]])

fig1, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig1.suptitle('B(H273Y) OST-treated ferrets')

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
fig2.suptitle('B(H273Y) placebo-treated ferrets')

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


# In[ ]:




