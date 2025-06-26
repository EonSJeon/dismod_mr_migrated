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

# print(mr_model.vars) # empty dict at the moment

mr_model.vars['p'] = result_dict['p']
# print(mr_model.vars)

print("==== [test5.py] fitting started ====")
map_estimate, idata = dismod_mr_pymc5.fit.asr(
    mr_model=mr_model,
    pm_model=pm_model,
    data_type='p',
    verbose=True,
    draws=1000,    # 테스트용: posterior 샘플 개수
    tune=500,      # 테스트용: tuning 단계
    chains=2,
    cores=4
)
print("==== [test5.py] fitting finished ====")

plt.figure(figsize=(10, 4))

dismod_mr_pymc5.plot.data_bars(
    mr_model.get_data('p'),
    color='grey',
    label='Simulated PD Data'
)


