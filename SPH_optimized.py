import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from scipy.integrate import RK23
import time

plt.rcParams.update({'font.size': 16})
plt.style.use('seaborn-whitegrid')

#_______________________________________ INITIAL CONDITIONS _______________________________________#
# Parameters
N = 400 # Number of particles 
mass_value = 0.001875 # Mass of the particles 
kappa = 2 # Used in the NNPS algorithm
nu = 1.4 # Scaling factor for the smoothing factor
gamma_sound = 1.4 # For the sound speed

mass = np.zeros(N) + mass_value

# Give the particles evenly spaced positions before and after the wall
x = np.zeros(N)
xs1 = np.linspace(-0.6, 0, 320, endpoint=False) # 320 left of the wall
xs2 = np.linspace(0, 0.6, 80, endpoint=False) + 0.0075 # 80 right of the wall
x[:len(xs1)] = xs1
x[-len(xs2):] = xs2

# Initial density for the particles
rho = np.zeros(N)
rho[:len(xs1)] = 1 # Density of the particles in the left
rho[-len(xs2):] = 0.25 # Density of the particles in the right

# Initial energy
e = np.zeros(N)
e[:len(xs1)] = 2.5 # Density of the particles in the left 
e[-len(xs2):] = 1.795 # Density of the particles in the right

# Initial velocities are the same, set at 0
v = np.zeros(N)

# Initial pressures-
p = np.zeros(N)
p[:len(xs1)] = 1 # Pressure of the particles in the left
p[-len(xs2):] = 0.1795 #  of the particles in the right

# Create the initial state vector as a matrix
W = np.zeros((N, 5)) # Change to W0.
W[:,0] = x
W[:,1] = rho
W[:,2] = v
W[:,3] = e
W[:,4] = p 

NParams = len(W[1])

# Reshape matrix into a vector
W = W.reshape(NParams*N)

#________________________________________ FUNCTIONS _______________________________________________#
def h_len(mass, density):
    """ Calculates the smoothing length for all the particles for a given
    state vector. """
    return np.zeros(N) + 0.005 #nu*(mass/density)

def smoothingW(dx, h):
    """ Utilizes the relative distances of a pair to calculate the 
    smoothing function. The input relative distanc has already been calculated
    with the smoothing factor h."""
    
    ad = (1/h) # Alpha-d factor
    R = abs(dx)/h #np.linalg.norm(dx)/h
    
    smoothW = np.zeros(len(R))
    
    # Define masks
    mask_01 = (R >= 0) & (R < 1) # First condition
    mask_02 = (R >= 1) & (R < 2) # Second condition
  
    # Optimize using masks and get rid of loop
    smoothW[mask_01] = ad[mask_01]*(2/3 - (R[mask_01])**2 + 0.5*(R[mask_01])**3)
    smoothW[mask_02] = ad[mask_02]*((2-(R[mask_02]))**3)/6   
                  
    return smoothW

def smoothingdW(dx, h):
    """ Utilizes the relative distances of a pair to calculate the 
    derivative of the smoothing function. The input relative distanc has 
    already been calculated with the smoothing factor h."""
    
    ad = (1/h) # Alpha-d factor
    R = abs(dx)/h #np.linalg.norm(dx)/h
    smoothdW = np.zeros(len(R)) # Variable length for the 3D case
    
    # Define masks
    mask_01 = (R >= 0) & (R < 1) # First condition
    mask_02 = (R >= 1) & (R < 2) # Second condition
  
    # Optimize using masks and get rid of loop
    smoothdW[mask_01] = ad[mask_01]*(-2 + 1.5*(R[mask_01]))*(dx[mask_01]/(h[mask_01])**2)
    smoothdW[mask_02] = -ad[mask_02]*(0.5*((2-(R[mask_02]))**2))*(dx[mask_02]/
            ((h[mask_02])*abs(dx[mask_02])))
    
    return smoothdW    

def artvisc(dx, rho_rel, dv, h, c_rel):
    """ Calculates the artificial viscosity for a pair. """
    
    # Define parameters that go into the artificial viscosity
    alpha = 1
    beta = 1

    # Relative quantities
    theta = 0.1*h
    phi_rel = (h*dv*dx)/(abs(dx)**2 + theta**2)  #(h*np.dot(dv, dx))/(abs(dx)**2 + theta**2) 
   
    # Calculate viscosity
    visc = (-alpha*c_rel*phi_rel + beta*phi_rel**2)/rho_rel
    
    return visc

def velocity(mass, density1, density2, pressure1, pressure2, smoothingdW, artvisc):
    """ Calculates the derivative of density for a given pair of particles. """
    return -mass*(pressure1/density1**2 + pressure2/density2**2 + artvisc)*smoothingdW
    
def energy(mass, density1, density2, pressure1, pressure2, vel1, vel2, smoothingdW, artvisc):
    return 0.5*mass*((pressure1/(density1**2)) + (pressure2/(density2**2)) 
                     + artvisc)*(vel1-vel2)*smoothingdW

#def density(mass, vel1, vel2, smoothingdW):
#    return mass*(vel1-vel2)*smoothingdW
    
#______________________________________ NNPS ALGORITHM ______ _____________________________________#

def NNPScalc(W):
    """ Nearest neightbour pair search algorithm - vectorized version. Structure follows the 
    for loop version. Calculates all the quantities involving pairs. 
   
    Legend for the output:
    NNPScalc(W)[0] = pi
    NNPScalc(W)[1] = pj
    NNPScalc(W)[2] = smoothW
    NNPScalc(W)[3] = smoothdW
    NNPScalc(W)[4] = viscosity """
    
    W = W.reshape(N, NParams) # Reshape vector into matrix.
    # Calculating the average smoothing length for ALL possible combinations of particles
    # we can easily pick the right one later one by knowning the indexes of the pairs.
    hmean = ((h_len(mass, W[:,1])).reshape(N, 1) + h_len(mass, W[:,1]))*0.5

    # Calculate dx and dv: use upper triangle of the matrix as the interactions are symmetric.
    dx = np.triu(W[:,0].reshape(len(W[:,0]), 1) - W[:,0])
    dv = np.triu(W[:,2].reshape(len(W[:,2]), 1) - W[:,2])
   
    # Find pairs and obtain indexes
    maskNNPS = (abs(dx) <= kappa*hmean) & (abs(dx) > 0)
    pi = np.where(maskNNPS)[0]
    pj = np.where(maskNNPS)[1]
    
    # Update difference vectors with the respective mask with the neighbors mask
    dx = dx[maskNNPS]
    dv = dv[maskNNPS]
    hmean = hmean[maskNNPS]
    
    # Calculate smoothing functions for each pair
    smoothW = smoothingW(dx, hmean)
    smoothdW = smoothingdW(dx, hmean)
    
    # Compute quantities from pair calculations
    # Sound speed
    c = np.sqrt((gamma_sound - 1)*W[:,3])
    cmean = np.triu((c.reshape(len(c), 1) + c))*0.5
    cmean = cmean[maskNNPS]
    
    # Mean density
    rhomean = np.triu((W[:,1]).reshape(N, 1) + W[:,1])*0.5
    rhomean = rhomean[maskNNPS]
    
    # Calculate artificial viscosity
#    maskVISC = (np.dot(dv,dx,  ) < 0) # Create mask for the artificial viscosity condition
    maskVISC = ((dv*dx) < 0)
    viscosity = np.zeros(len(pi))
    viscosity[maskVISC] = artvisc(dx[maskVISC], rhomean[maskVISC],
             dv[maskVISC], hmean[maskVISC], cmean[maskVISC])
    
    return pi, pj, smoothW, smoothdW, viscosity
   
def integrate(t, W):
    """ Function to integrate. """
    
    # Extract information related to the nearest neighbors
    pi_s, pj_s, smoothW, smoothdW, artvisc = NNPScalc(W)
    npairs = len(pi_s)
    
    # Calculate density using the summation density
    W = W.reshape(N, NParams)
   
    W[:,1] = mass*(2/(3*(h_len(mass, W[:,1])))) # Density self effect (for every particle)
    
    dW = np.zeros(np.shape(W)) # Empty array to store the derivatives

    for k in range(npairs):        
        pi = pi_s[k]
        pj = pj_s[k]
        
        W[pi,1] += mass[pj]*smoothW[k]
        W[pj,1] += mass[pi]*smoothW[k]
    
    gamma_pressure = 1.4
    W[:,4]  = (gamma_pressure - 1)*(W[:,1]*W[:,3]) # Updates pressure. Depends only on particle
    
    # Compute the derivatives      
    for k in range(npairs):        
        #W[:,0] = x
        #W[:,1] = rho
        #W[:,2] = v
        #W[:,3] = e
        #W[:,4] = p 
    
        # Get index for each particle of the pair
        pi = pi_s[k]
        pj = pj_s[k]
        
        # Compute (derivatives of) velocities for each particle from the pair
        dW[pi,2] += velocity(mass[pj], W[pi,1], W[pj,1], W[pi,4], W[pj,4], smoothdW[k], artvisc[k])
        dW[pj,2] += -velocity(mass[pi], W[pj,1], W[pi,1], W[pj,4], W[pi,4], smoothdW[k], artvisc[k])
        
        # Derivatives of the internal energy
        dW[pi,3] += energy(mass[pj], W[pi,1], W[pj,1], W[pi,4], W[pj,4], W[pi,2], W[pj,2], smoothdW[k], artvisc[k])
        dW[pj,3] -= energy(mass[pi], W[pj,1], W[pi,1], W[pj,4], W[pi,4], W[pj,2], W[pi,2], smoothdW[k], artvisc[k])
      
    # Derivatives of density and pressure are 0     
    dW[:,1] = 0 
    dW[:,4] = 0
    dW[:,0] = W[:,2] # Derivative of the position is the input velocity
    
    dW = dW.reshape(N*NParams)
    
    return dW
    
#_________________________________________ INTEGRATION ____________________________________________#
    
start_time = time.time()
# Integration parameters
tstep = 0.005
tmin = 0
tmax = tstep*40
steps = np.arange(tstep, tmax, tstep)
NSteps = len(steps)

# Integrator setup
W_int = RK45(integrate, 0.005, W, 40, max_step = 0.05) #, rtol=10e-9, atol=10-9)
W_i = np.zeros([NSteps, N, NParams])

# Loop for the integration
for i in range(NSteps):
    W_i[i] = np.array(W_int.y).reshape(N, NParams) # Select current state, reshape and store
    W_int.step()        
    print(i)
    
#___________________________________________ PLOTTING _____________________________________________#
# Define the state vector for the last integrated timestep.
W_plot = W_i[34]    
    
# Plot densities
plt.figure(figsize=[10,8])
plt.plot(W_plot[:,0], W_plot[:,1], '-.k')
plt.title('Density')
plt.xlabel('$Position \, \,[m]$')
plt.xlim([-0.4, 0.4])
plt.ylabel('$Density \, \, [kg/m^{3}]$')
plt.savefig('density1d-35.png', dpi=300,  bbox_inches='tight')

# Plot velocities
plt.figure(figsize=[10,8])
plt.plot(W_plot[:,0], W_plot[:,2], '-.k')
plt.title('Velocity')
plt.xlabel('$Position \, \,[m]$')
plt.xlim([-0.4, 0.4])
plt.ylim([0, 1])
plt.ylabel('$Velocity \, [m/s]$')
plt.savefig('vel1d-35.png', dpi=300,  bbox_inches='tight')

# Plot energies
plt.figure(figsize=[10,8])
plt.plot(W_plot[:,0], W_plot[:,3], '-.k')
plt.title('Energy')
plt.xlabel('$Position \, \,[m]$')
plt.xlim([-0.4, 0.4])
plt.ylim([1.6, 2.6]) 
plt.ylabel('$Internal \, \, energy \, \, [J/kg]$')
plt.savefig('energy1d-35.png', dpi=300,  bbox_inches='tight')

# Plot pressures
plt.figure(figsize=[10,8])
plt.plot(W_plot[:,0], W_plot[:,4], '-.k')
plt.xlim([-0.4, 0.4])
plt.ylim([0, 1.2]) 
plt.title('Pressure ')
plt.xlabel('$Position \, \,[m]$')
plt.ylabel('$Pressure \, \, [N/m^{2}]$')
plt.savefig('pressure1d-35.png', dpi=300,  bbox_inches='tight')


print("--- %s seconds ---" % (time.time() - start_time))
#    
#    
#fig = plt.figure()
#ax = fig.add_subplot((111))#, projection='3d')
#for i in range(39):
#    ax.plot(W_i[i,:,0], W_i[i,:,2]) 
#    ax.set_xlim([-0.4, 0.4])
#    ax.set_ylim([0, 1])
#    plt.tight_layout()
#    plt.pause(0.1)
#    ax.clear()
#    

# Plots for the report

    
# Plot of the initial densities
plt.figure(figsize=[10,6])
plt.scatter(W[:,0], W[:,1], s=0.5, c='k') #, '-.k')
plt.vlines(0, 0, 1.25, color='red', linestyle = '-.', label='$Diaphragm$')
plt.title('Initial schema')
plt.xlabel('$Position \, \,[m]$')
plt.xlim([-0.65, 0.65])
plt.ylabel('$Density \, \, [kg/m^{3}]$')
plt.legend(frameon=True)
plt.savefig('schema.png', dpi=300,  bbox_inches='tight')


    
    
    
    
    
