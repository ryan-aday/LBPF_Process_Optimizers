import numpy as np
from scipy.optimize import minimize

# Material properties (example values)
material_properties = {
    'thermal_conductivity': 205,  # W/mK
    'melting_point': 660,  # Celsius
    'density': 2700,  # kg/m³
    'specific_heat': 897,  # J/kgK
    'thermal_diffusivity': 5.38e-6,  # m²/s
    'thermal_expansion': 23.1e-6,  # 1/K
    'modulus_of_elasticity': 70e9,  # Pa
    'enthalpy': 1.2e6,  # J/kg
    'absorptivity': 0.68,
    'penetration_depth': 0.35,  # mm
    'threshold_energy_density': 8.4  # N/mm
}

# Laser properties (assumed constants)
laser_properties = {
    'beam_diameter': 0.5  # mm
}

# Target normalized enthalpy
target_normalized_enthalpy = 30

# Objective function to minimize
def objective(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params

    # Calculate normalized enthalpy
    normalized_enthalpy = (material_properties['absorptivity'] * laser_power) / (
        material_properties['density'] * material_properties['enthalpy'] *
        np.sqrt(np.pi * material_properties['thermal_diffusivity'] * scan_speed * laser_properties['beam_diameter'])
    )

    # Calculate deviation from target normalized enthalpy
    deviation_normalized_enthalpy = abs(normalized_enthalpy - target_normalized_enthalpy)

    return deviation_normalized_enthalpy

# Melt pool depth calculation using Beer-Lambert Law
def melt_pool_depth(P, A, v, d, z):
    rho = material_properties['density']
    c_p = material_properties['specific_heat']
    E_0 = P / A
    return E_0 * np.exp(-z / material_properties['penetration_depth']) - material_properties['threshold_energy_density']

# Melt pool width calculation
def melt_pool_width(P, v):
    rho = material_properties['density']
    c_p = material_properties['specific_heat']
    return 2 * np.sqrt(P / (rho * c_p * v))

# Constraints
def constraint_melt_pool_depth(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params
    A = np.pi * (laser_properties['beam_diameter'] / 2) ** 2
    depth = melt_pool_depth(laser_power, A, scan_speed, laser_properties['beam_diameter'], layer_thickness)
    return depth - layer_thickness  # e >= t_l

def constraint_hatch_distance(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params
    width = melt_pool_width(laser_power, scan_speed)
    return hatch_spacing - width

# Initial guesses for parameters
initial_params = [200, 1000, 0.1, 0.03]  # Example values: laser_power (W), scan_speed (mm/s), hatch_spacing (mm), layer_thickness (mm)

# Parameter bounds
bounds = [
    (100, 500),  # Laser power range (W)
    (500, 2000),  # Scan speed range (mm/s)
    (0.05, 0.2),  # Hatch spacing range (mm)
    (0.01, 0.1)  # Layer thickness range (mm)
]

# Constraints in dictionary format
constraints = [
    {'type': 'ineq', 'fun': constraint_melt_pool_depth},
    {'type': 'ineq', 'fun': constraint_hatch_distance}
]

# Optimization
result = minimize(objective, initial_params, bounds=bounds, constraints=constraints, method='L-BFGS-B')

# Optimal parameters
optimal_params = result.x
laser_power_opt, scan_speed_opt, hatch_spacing_opt, layer_thickness_opt = optimal_params

print(f"Optimal Laser Power: {laser_power_opt:.2f} W")
print(f"Optimal Scan Speed: {scan_speed_opt:.2f} mm/s")
print(f"Optimal Hatch Spacing: {hatch_spacing_opt:.2f} mm")
print(f"Optimal Layer Thickness: {layer_thickness_opt:.2f} mm")

# Use the optimized parameters in the LPBF process setup
