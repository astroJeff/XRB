# Constants to be used throughout module
G = 6.674e-8 # Gravitational constant in cgs
GGG = 1.909e5 # Gravitational constant in Rsun * (km/s)^2 / Msun
c_light = 2.9979e10 # speed of light in cgs
km_to_cm = 1.0e5 # km to cm
Msun_to_g = 1.989e33 # Msun to g
Rsun_to_cm = 6.995e10 # Rsun to cm
AU_to_cm = 1.496e13 # AU to cm
pc_to_cm = 3.086e18 # parsec to cm
pc_to_km = 3.086e13 # parsec to km
yr_to_sec = 31557600.0 # Sec in yr
day_to_sec = 3600.0*24.0 # Sec in day
deg_to_rad = 0.0174532925199 # Degrees to radians
rad_to_deg = 57.2957795131 # Radians to degrees
asec_to_rad = 4.84814e-6 # Arcsec to radians


M_0 = 0.073 # in Msun
xi = 0.704
M_NS = 1.35
#v_wind = 1000.0 # companion velocity in km/s
R_NS = 10.0  # NS radius in km
eta_bol = 0.15  # Calibrated to Chandra X-ray sensitivity
v_k_sigma = 265.0  # Kick velocity Maxwellian dispersion - Fe-core collapse
v_k_sigma_ECS = 50.0  # Kick velocity Maxwellian dispersion - ECS
ECS_Fe_mass = 10.0  # Mass differentiating ECS from Fe-core collapse
alpha = -2.35  # IMF index


# Parameter ranges
min_mass = 8.0
max_mass = 39.0
min_A = 1.0e1
max_A = 1.0e4
min_z = 0.001
max_z = 0.03



###################  STAR FORMATION HISTORY SCHEME #################
sf_scheme = None


dist_LMC = 5.0e4 * pc_to_km # Distance to Large Magellanic Cloud (in km)
dist_SMC = 6.1e4 * pc_to_km # Distance to Small Magellanic Cloud (in km)
dist_NGC4244 = 4.3e6 * pc_to_km # Distance to NGC4244 (in km)
