import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.getcwd()))

import dismod_mr_pymc5

mr_model = dismod_mr_pymc5.data.load('amd_sim_data_Intermediate')

pm_model, result_dict = dismod_mr_pymc5.model.asr(
    mr_model,
    'p',     
)

print("pm_model:")
print(pm_model)
print()

print("result_dict:")
print(result_dict['p'].keys())
print()