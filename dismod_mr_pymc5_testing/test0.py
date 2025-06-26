import pymc as pm

with pm.Model() as pm_model:
    pass

print("Available model attributes:")
model_attrs = [attr for attr in dir(pm_model) if not attr.startswith('_')]
for attr in sorted(model_attrs):
    print(f"   - {attr}")
print()

# Available model attributes:
#    - RV_dims
#    - add_coord
#    - add_coords
#    - add_named_variable
#    - basic_RVs
#    - check_bounds
#    - check_start_vals
#    - compile_d2logp
#    - compile_dlogp
#    - compile_fn
#    - compile_logp
#    - cont_vars
#    - contexts
#    - continuous_value_vars
#    - coords
#    - create_value_var
#    - d2logp
#    - datalogp
#    - debug
#    - deterministics
#    - dim_lengths
#    - disc_vars
#    - discrete_value_vars
#    - dlogp
#    - eval_rv_shapes
#    - free_RVs
#    - initial_point
#    - initial_values
#    - isroot
#    - logp
#    - logp_dlogp_function
#    - make_obs_var
#    - model
#    - name
#    - name_for
#    - name_of
#    - named_vars
#    - named_vars_to_dims
#    - observed_RVs
#    - observedlogp
#    - parent
#    - point_logps
#    - potentiallogp
#    - potentials
#    - prefix
#    - profile
#    - register_rv
#    - replace_rvs_by_values
#    - root
#    - rvs_to_initial_values
#    - rvs_to_transforms
#    - rvs_to_values
#    - set_data
#    - set_dim
#    - set_initval
#    - shape_from_dims
#    - str_repr
#    - unobserved_RVs
#    - unobserved_value_vars
#    - update_start_vals
#    - value_vars
#    - values_to_rvs
#    - varlogp
#    - varlogp_nojac

with pm.Model() as pm_model:
    alpha = pm.Normal('Alpha', mu=0, sigma=1)
    pi = pm.Constant('pi', 3.14) # constant!!
    half_pi = pm.Deterministic('half_pi', 3.14 / 2) # deterministic!!

print("Available model attributes + alpha, pi, half_pi")
model_attrs = [attr for attr in dir(pm_model) if not attr.startswith('_')]
for attr in sorted(model_attrs):
    print(f"   - {attr}")
print()

print("=== MODEL VARIABLES ANALYSIS ===")
print(f"named_vars: {pm_model.named_vars}")
print(f"free_RVs: {pm_model.free_RVs}")
print(f"observed_RVs: {pm_model.observed_RVs}")
print(f"deterministics: {pm_model.deterministics}")
print(f"unobserved_RVs: {pm_model.unobserved_RVs}")
print(f"unobserved_value_vars: {pm_model.unobserved_value_vars}")
print(f"value_vars: {pm_model.value_vars}")
print(f"rvs_to_values: {pm_model.rvs_to_values}")
print(f"rvs_to_initial_values: {pm_model.rvs_to_initial_values}")
print(f"rvs_to_transforms: {pm_model.rvs_to_transforms}")
print()

print("=== CONSTANT AND DETERMINISTIC ANALYSIS ===")
print(f"pi type: {type(pi)}")
print(f"pi value: {pi}")
print(f"half_pi type: {type(half_pi)}")
print(f"half_pi value: {half_pi}")
print()

print("=== CHECKING IF THEY APPEAR IN MODEL ATTRIBUTES ===")
print(f"Is pi in named_vars? {'pi' in pm_model.named_vars}")
print(f"Is half_pi in named_vars? {'half_pi' in pm_model.named_vars}")
print(f"Is pi in deterministics? {pi in pm_model.deterministics}")
print(f"Is half_pi in deterministics? {half_pi in pm_model.deterministics}")
print(f"Is pi in free_RVs? {pi in pm_model.free_RVs}")
print(f"Is half_pi in free_RVs? {half_pi in pm_model.free_RVs}")
print()

print("=== TRYING TO ACCESS CONSTANTS ===")
try:
    print(f"pm_model.pi: {pm_model.pi}")
except AttributeError as e:
    print(f"pm_model.pi: AttributeError - {e}")

try:
    print(f"pm_model.half_pi: {pm_model.half_pi}")
except AttributeError as e:
    print(f"pm_model.half_pi: AttributeError - {e}")