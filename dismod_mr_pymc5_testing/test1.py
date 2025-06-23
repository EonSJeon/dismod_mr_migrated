import sys
import os

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.getcwd()))

import dismod_mr_pymc5

mr_model = dismod_mr_pymc5.data.load('amd_sim_data_Intermediate')

mr_model.describe('p')

mr_model.keep(areas=['Global'], sexes=['Female', 'Male', 'Both'])

mr_model.describe('p')

mr_model.keep(areas=['United States of America'], sexes=['Female'])

mr_model.describe('p')


