import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.getcwd()))

import dismod_mr_pymc5

mr_model = dismod_mr_pymc5.data.load('amd_sim_data_Intermediate')

print(mr_model.parameters['p'])
print()

print(mr_model.parameters.get('p', {}))
print()

parameters = mr_model.parameters
p_parameters = mr_model.parameters['p']

print(p_parameters['parameter_age_mesh'])
print()

print(p_parameters['smoothness'])
print()

print(parameters['ages'])
print()

print(p_parameters['ages'])
print()



