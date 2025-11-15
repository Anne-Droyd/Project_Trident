#conversion of planetary data to usable parameters

#mercury, assume liquid core mantle and atmosphere
import numpy as np
outer_radius = 2440000
inner_radius = 2020000
density_mantle = 3650
mass_mercury = 3.3011e23

volume_out = 4/3*np.pi*outer_radius**3
volume_in = 4/3*np.pi*inner_radius**3
volume = volume_out-volume_in
mass = density_mantle*volume/mass_mercury
print(mass)

#venus
inner_radius = 3147000
outer_radius = 6051800
venus_mass = 4.867e24
venus_atmosphere = 4.8e20/venus_mass
#assume earth core density
core_density = 13000
#assume earth mantle density

mass = core_density*4/3*np.pi*inner_radius**3/venus_mass
print(mass, 1-mass-venus_atmosphere, venus_atmosphere)

#moon
atm = 2.5e16
mars_mass = 6.39e23
atm_mass = atm/mars_mass
print(1-0.24-atm_mass)

#jupiter
m_core=18/318 #earth mass -> mass frac
print(m_core)
Z = 0.08
H_he= (1-Z)

#saturn
mass=95.16 #earth mass
mcore=13.5/mass #mass_frac
print(mcore)
mass_z = 27.5/mass #earth mass -> mass frac
print(mass_z)
print(1-mcore-mass_z)

#uranus
sio =  11.3
h20 = 13.2

#neptune
sio = 13.1
h20 = 15.4
