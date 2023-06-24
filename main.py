'''Imports'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

'''Configuration'''

# Size of the simulation box
window = (20, 20)
# Resolution of the electric field
field_resolution = 25 # Default of 25

# Resolution of the potential map
potential_resolution=250 # Default of 250

# Coulomb's Constant, in real life it is 1/4pie0
const = 1

'''Functions'''

# Multipole Creator
def multipole(n,r,charge):
    # Sample n points around a circle of radius r
    theta = np.radians(np.linspace(0,360,n+1)[:-1])
    
    # X positions
    x = r * np.sin(theta)
    # Y positions
    y = r * np.cos(theta)

    # Alternating positive and negative charges
    q = [-charge if cat & 0x1 else charge for cat in list(range(n))]

    return [(X,Y,Q) for X,Y,Q in zip(x,y,q)]

# Electric Field
def E(x,y,particles):
    # Charge of the particles
    charge = [p[2] for p in particles]

    # Separation vector from each particle to each field point
    # Make one numpy array for each particle
    # One entry in the list per particle. Each entry is a tuple of x and y components of the separation vector for each point in the field
    particle_separation_vectors = [(x-p[0],y-p[1]) for p in particles]
    
    # Calculate Distances between field point and source point
    distances = [np.sqrt(p[0]**2 + p[1]**2) for p in particle_separation_vectors]

    # Unit Vector Array
    unit_vector = [(np.divide(p[0],d,out=np.zeros_like(p[0]),where=d!=0),np.divide(p[1],d,out=np.zeros_like(p[1]),where=d!=0)) for p,d in zip(particle_separation_vectors,distances)]

    # Calculate electric field at each point, due to each particle
    field_x = const * sum([np.divide(q*p[0],(d**2),out=np.zeros_like(q*p[0]),where=d**2!=0) for p,d,q in zip(unit_vector,distances,charge)])
    field_y = const * sum([np.divide(q*p[1],(d**2),out=np.zeros_like(q*p[1]),where=d**2!=0) for p,d,q in zip(unit_vector,distances,charge)])
    
    # Calculate the magnitude of the electric field at each point
    magnitude = np.sqrt((field_x**2 + field_y**2))

    # Return unit vectors for the field and the magnitude
    return field_x/magnitude, field_y/magnitude, magnitude

# Electric Potential
def V(x,y,particles):
    # Charge of the particles
    charge = [p[2] for p in particles]

    # Separation vector from each particle to each field point
    # Make one numpy array for each particle
    # One entry in the list per particle. Each entry is a tuple of x and y components of the separation vector for each point in the field
    particle_separation_vectors = [(x-p[0],y-p[1]) for p in particles]
    
    # Calculate Distances between field point and source point
    distances = [np.sqrt(p[0]**2 + p[1]**2) for p in particle_separation_vectors]

    # Calculate electric potential map
    potential = const * sum([q/d for d,q in zip(distances,charge)])

    # Return potential map
    return potential

'''Initialization'''
# Initiate set of particles in a multipole configuration
particles = multipole(8,2,1)

'''This script by default uses a multipole configuration with 8 alternating 1C charges in a circle of radius 2.
However, any particle configuration can be specified by uncommenting the code below'''

##particles = [(x1,y1,q1),(x2,y2,q2),(x3,y3,q3)]

# List of each particle's position
px = [p[0] for p in particles]
py = [p[1] for p in particles]

# Meshgrid for vectors
x,y = np.meshgrid(np.linspace(-1*window[0]/2,window[0]/2,field_resolution),np.linspace(-1*window[1]/2,window[1]/2,field_resolution))

# Meshgrid for potential
hires_x, hires_y = np.meshgrid(np.linspace(-1*window[0]/2,window[0]/2,potential_resolution),np.linspace(-1*window[1]/2,window[1]/2,potential_resolution))

'''Calculation of field and potential'''

# Potential map
pot = V(hires_x,hires_y,particles)

# Field map
fx,fy,mag = E(x,y,particles)

# Mask out zero length vectors
mask = mag==0

'''Matplotlib Implementation'''

# Create plot
fig, ax = plt.subplots()

# Plot potential scalar field
ax.scatter(hires_x,hires_y,c=pot,cmap='seismic',norm=colors.SymLogNorm(linthresh=0.03,vmin=pot.min(),vmax=pot.max()))

# Plot electric field
ax.quiver(x[~mask],y[~mask],fx[~mask],fy[~mask],mag[~mask],cmap='Blues',norm=colors.LogNorm(vmin=mag[~mask].min(),vmax=mag[~mask].max()))

# Plot points
ax.scatter(px,py,c="green")

# Set plot to be the correct aspect ratio
ax.set_aspect('equal', adjustable='box')

# Show plot
plt.show()