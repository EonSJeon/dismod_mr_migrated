import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.getcwd()))

import dismod_mr_pymc5

mr_model = dismod_mr_pymc5.data.load('amd_sim_data_Intermediate')

summary = mr_model.input_data.groupby('data_type')['value'].describe()
print(np.round(summary,3).sort_values('count', ascending=False))
print()

groups = mr_model.get_data('p').groupby('area')
print(np.round_(groups['value'].describe(),3).sort_values('50%', ascending=False))
print()

countries = ['United States of America']
c = {}
for i, c_i in enumerate(countries):
    c[i] = groups.get_group(c_i)
    
ax = None
plt.figure(figsize=(10,4))
for i, c_i in enumerate(countries):
    ax = plt.subplot(1,2,1+i, sharey=ax, sharex=ax)
    dismod_mr_pymc5.plot.data_bars(c[i]) # TODO
    plt.xlabel('Age (years)')
    plt.ylabel('Prevalence (per 1)')
    plt.title(c_i)
plt.axis(ymin=-.001, xmin=-5, xmax=105)
plt.subplots_adjust(wspace=.3)
plt.show()