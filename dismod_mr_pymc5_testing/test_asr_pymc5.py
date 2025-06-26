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


# Inspect random variables in the PyMC model (PyMC 5 API)
print("=== RANDOM VARIABLES IN PM_MODEL (PyMC 5) ===")
print()

# Method 1: Get all variables in the model
print("1. All variables in the model:")
try:
    # PyMC 5: model.vars contains all variables
    for var in pm_model.vars:
        print(f"  - {var.name}: {type(var).__name__}")
except AttributeError:
    print("  Model doesn't have .vars attribute")
print()

# Method 2: Get variables by type using PyMC 5 API
print("2. Variables by type:")
try:
    # PyMC 5: model.free_RVs contains unobserved random variables
    print("   Free random variables (unobserved):")
    for var in pm_model.free_RVs:
        print(f"     - {var.name}: {type(var).__name__}")
    print()
    
    # PyMC 5: model.observed_RVs contains observed random variables
    print("   Observed random variables:")
    for var in pm_model.observed_RVs:
        print(f"     - {var.name}: {type(var).__name__}")
    print()
    
    # PyMC 5: model.deterministics contains deterministic variables
    print("   Deterministic variables:")
    for var in pm_model.deterministics:
        print(f"     - {var.name}: {type(var).__name__}")
    print()
    
except AttributeError as e:
    print(f"  Error accessing model attributes: {e}")
print()

# Method 3: Get variables as a dictionary
print("3. Variables as dictionary:")
try:
    # PyMC 5: model.named_vars contains all named variables
    for name, var in pm_model.named_vars.items():
        print(f"  - {name}: {type(var).__name__}")
except AttributeError:
    print("  Model doesn't have .named_vars attribute")
print()

# Method 4: Get model graph structure
print("4. Model graph structure:")
try:
    print(f"   Number of free random variables: {len(pm_model.free_RVs)}")
    print(f"   Number of observed random variables: {len(pm_model.observed_RVs)}")
    print(f"   Number of deterministic variables: {len(pm_model.deterministics)}")
    print(f"   Total variables: {len(pm_model.vars) if hasattr(pm_model, 'vars') else 'unknown'}")
except AttributeError as e:
    print(f"   Error getting model structure: {e}")
print()

# Method 5: Alternative way to get all variables
print("5. Alternative inspection methods:")
try:
    # Get all variables using model.graph
    print("   All variables from model.graph:")
    for var in pm_model.graph.vars:
        print(f"     - {var.name}: {type(var).__name__}")
except AttributeError:
    print("   Model doesn't have .graph.vars attribute")
print()

# Method 6: Check what attributes the model has
print("6. Available model attributes:")
model_attrs = [attr for attr in dir(pm_model) if not attr.startswith('_')]
for attr in sorted(model_attrs):
    print(f"   - {attr}")
print()