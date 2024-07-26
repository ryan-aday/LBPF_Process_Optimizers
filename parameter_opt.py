import numpy as np
from scipy.optimize import minimize

# Material properties (example values)
material_properties = {
    'thermal_conductivity': 205,  # W/mK
    'melting_point': 660,  # Celsius
    'initial_temp': 25,  # Celsius
    'pouring_temp': 700,  # Celsius
    'density': 2700,  # kg/m³
    'specific_heat_solid': 897,  # J/kgK
    'specific_heat_liquid': 1000,  # J/kgK, assumed
    'heat_of_fusion': 397,  # kJ/kg, assumed
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

# Constraint targets (modifiable)
target_normalized_enthalpy = 30
target_solubility = 1  # Adjust as needed
target_shrinkage = 0.02  # Adjust as needed

# Range of scan lengths and widths (example values)
scan_lengths = np.linspace(100, 200, 5)  # mm
scan_widths = np.linspace(10, 50, 5)  # mm

# Objective function to minimize
def objective(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params

    total_combined_objective = 0
    for scan_length in scan_lengths:
        for scan_width in scan_widths:
            # Calculate power density
            power_density = laser_power / (np.pi * (laser_properties['beam_diameter'] / 2) ** 2)

            # Calculate time to receive sufficient curing energy
            t1 = material_properties['threshold_energy_density'] / power_density

            # Calculate maximum feed rate
            max_feed_rate = (0.866 * laser_properties['beam_diameter']) / t1

            # Calculate time to cure one straight line
            t2 = scan_length / max_feed_rate

            # Calculate time to cure entire surface
            num_scans = np.ceil(scan_width / (laser_properties['beam_diameter'] / 2))
            total_time = t2 * num_scans

            # Add total time for this combination to the overall objective
            total_combined_objective += total_time

    return total_combined_objective

# Melt pool depth calculation using Beer-Lambert Law
def melt_pool_depth(P, A, z):
    E_0 = P / A
    return E_0 * np.exp(-z / material_properties['penetration_depth']) - material_properties['threshold_energy_density']

# Melt pool width calculation
def melt_pool_width(P, v):
    rho = material_properties['density']
    c_p = material_properties['specific_heat_solid']
    return 2 * np.sqrt(P / (rho * c_p * v))

# Paraboloid volume calculation
def paraboloid_volume(d, e):
    r = d / 2
    return (1/2) * np.pi * r**2 * e

# Constraint for normalized enthalpy
def constraint_normalized_enthalpy(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params

    # Calculate normalized enthalpy
    normalized_enthalpy = (material_properties['absorptivity'] * laser_power) / (
        material_properties['density'] * material_properties['enthalpy'] *
        np.sqrt(np.pi * material_properties['thermal_diffusivity'] * scan_speed * laser_properties['beam_diameter'])
    )

    return normalized_enthalpy - target_normalized_enthalpy

# Additional constraint for minimum laser power based on the provided slides
def constraint_min_laser_power(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params
    d = laser_properties['beam_diameter']
    e = melt_pool_depth(laser_power, np.pi * (d / 2) ** 2, layer_thickness)
    V = paraboloid_volume(d, e)
    min_laser_power = material_properties['density'] * V * (
        material_properties['specific_heat_solid'] * (material_properties['melting_point'] - material_properties['initial_temp']) +
        material_properties['heat_of_fusion'] * 1e3 +  # converting kJ/kg to J/kg
        material_properties['specific_heat_liquid'] * (material_properties['pouring_temp'] - material_properties['melting_point'])
    )
    return laser_power - min_laser_power

# Constraint for solubility
def constraint_solubility(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params

    # Calculate power density
    power_density = laser_power / (np.pi * (laser_properties['beam_diameter'] / 2) ** 2)

    # Solubility: Lower solubility is better
    solubility = power_density / (scan_speed * layer_thickness)  # Simplified representation

    # Return solubility as a constraint
    return target_solubility - solubility

# Constraint for shrinkage
def constraint_shrinkage(params):
    laser_power, scan_speed, hatch_spacing, layer_thickness = params

    # Calculate expected shrinkage based on thermal expansion
    shrinkage = material_properties['thermal_expansion'] * (material_properties['melting_point'] - material_properties['initial_temp'])

    # Return shrinkage as a constraint
    return target_shrinkage - shrinkage

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
    {'type': 'eq', 'fun': constraint_normalized_enthalpy},
    {'type': 'ineq', 'fun': constraint_min_laser_power},
    {'type': 'ineq', 'fun': constraint_solubility},
    {'type': 'ineq', 'fun': constraint_shrinkage}
]

# Optimization using SLSQP
result = minimize(objective, initial_params, bounds=bounds, constraints=constraints, method='SLSQP')

# Optimal parameters
optimal_params = result.x
laser_power_opt, scan_speed_opt, hatch_spacing_opt, layer_thickness_opt = optimal_params

# Calculate and print the cure time for the optimized parameters
total_cure_time = 0
for scan_length in scan_lengths:
    for scan_width in scan_widths:
        power_density = laser_power_opt / (np.pi * (laser_properties['beam_diameter'] / 2) ** 2)
        t1 = material_properties['threshold_energy_density'] / power_density
        max_feed_rate = (0.866 * laser_properties['beam_diameter']) / t1
        t2 = scan_length / max_feed_rate
        num_scans = np.ceil(scan_width / (laser_properties['beam_diameter'] / 2))
        total_time = t2 * num_scans
        total_cure_time += total_time

print(f"Optimal Laser Power: {laser_power_opt:.2f} W")
print(f"Optimal Scan Speed: {scan_speed_opt:.2f} mm/s")
print(f"Optimal Hatch Spacing: {hatch_spacing_opt:.2f} mm")
print(f"Optimal Layer Thickness: {layer_thickness_opt:.2f} mm")
print(f"Total Cure Time for Optimized Parameters: {total_cure_time:.2f} seconds")

# Use the optimized parameters in the LPBF process setup
