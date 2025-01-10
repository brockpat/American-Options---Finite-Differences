#%% libraries
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.optimize
from scipy.stats import norm

path = " "

#%% Define Parameters as globals

#Parameters of GBM for Stock
mu = 0.06 #expected risky return
r = 0.02 #risky return
sigma = 0.2 #volatility

#Time Horizon
T=1

#Time Grid
N_t = 50 #Number of Gridpoints
t_grid = np.linspace(T,0,N_t+1) #time Grid

#Option Parameters
option_type = 'Put' #Option Type
S_0 = 10  #Initial Stock Price
K   = S_0 #Strike Price - Boundary conditions are calibrated to match at the money option. 
#Changing K will result in poorer performance of the Implicit Method
#%% Define Option Payoff
def payoff(S,K,option_type):
    if option_type == 'Call':
        return S-K
    if option_type == 'Put':
        return K-S
    else:
        raise ValueError('Select Call or Put as an Option Type.')

#%% Simulate Stock Price

#Set seed to allow replication
np.random.seed(48900531)

#Simulate Stock Price
def simulate_price(drift, sigma, t_grid, S_0, N_paths):
    # Initialize a matrix to store all paths column-wise
    S_mat = np.zeros((N_t + 1, N_paths))
    # Set the initial price for all paths
    S_mat[0, :] = S_0  
    #Simulate Brownian increments
    dt = t_grid[0]-t_grid[1]
    dW_t = np.random.normal(0, scale=np.sqrt(dt), size=(N_t,N_paths))  

    # Generate paths
    for i in range(N_paths):        
        for t in range(1, N_t+1):
            dS = drift * S_mat[t-1, i] * dt + sigma * S_mat[t-1, i] * dW_t[t-1,i]
            S_mat[t, i] = S_mat[t-1, i] + dS
    
    return S_mat

#Simulate Stock price paths under physical measure
S_mat = simulate_price(mu,sigma,t_grid,S_0,N_paths = 500)
    
# Plot all paths
for i in range(0,len(S_mat[0,:])):
    plt.plot(t_grid[::-1], S_mat[:, i])
plt.title('Simulated Price Paths')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.savefig(path + "Figures/Simulation.pdf",dpi = 600, bbox_inches='tight')
plt.show()
#%% Build Stock Price Grid

#Set Lower and Upper Bound of Grid 
S_min = np.percentile(S_mat[-1,:],1) 
S_max = np.percentile(S_mat[-1,:],99)

#Build uniformly spaced grid
S_grid = np.linspace(S_min,S_max,200) # Stock Price Grid with 200 gridpoints

#%% Initialise terminal condition
V_terminal = np.maximum(payoff(S_grid,K, option_type),0)

#%% PDE Solution Methods

#---- Coefficients alpha, beta, gamma
def compute_coefficients(S_grid,t_grid):
    N_S = len(S_grid)
    Delta_S = S_grid[1]-S_grid[0]
    Delta_t = t_grid[0]-t_grid[1]
    # Coefficients for the tridiagonal system
    alpha = np.zeros(N_S)
    beta = np.zeros(N_S)
    gamma = np.zeros(N_S)

    for i in range(1, N_S - 1):
        S_i = S_grid[i]
        alpha[i] = (-0.5 * sigma**2 * S_i**2 * Delta_t / (Delta_S**2)) + (0.5 * r * S_i * Delta_t / Delta_S)
        beta[i] = 1 + r * Delta_t + (sigma**2 * S_i**2 * Delta_t / (Delta_S**2))
        gamma[i] = (-0.5 * sigma**2 * S_i**2 * Delta_t / (Delta_S**2)) - (0.5 * r * S_i * Delta_t / Delta_S)

    return alpha,beta,gamma,Delta_S,Delta_t

#---- Implicit Method with Boundary Conditions
def implicit_method(Vn1,Vn,alpha,beta,gamma):    
    loss = [np.maximum(payoff(S_grid[0],K,option_type),0)-Vn1[0]]
    
    for i in range(1,len(S_grid)-1):
        loss.append(alpha[i]*Vn1[i-1] + beta[i]*Vn1[i] + gamma[i]*Vn1[i+1] - Vn[i])
    
    
    loss.append(np.maximum(payoff(S_grid[-1],K,option_type),0) - Vn1[-1])
    
    return np.array(loss)

#---- Implicit Method WITHOUT Boundary Conditions
def implicit_method_noBoundary(Vn1,Vn,alpha,beta,gamma,Delta_S,Delta_t):
    S_0 = S_grid[0]
    S_N = S_grid[-1]
    
    #Loss at first gridpoint (Forward Differences Only)
    loss = [((Vn1[0] - Vn[0]) / Delta_t 
             - 0.5 * sigma**2 * S_0**2 * (Vn1[2] - 2 * Vn1[1] + Vn1[0]) / (Delta_S**2)
             - r * S_0 * (Vn1[1] - Vn1[0]) / Delta_S 
             + r * Vn1[0])  
            ]
    
    #Interior Gridpoints Loss
    for i in range(1,len(S_grid)-1):
        loss.append(alpha[i]*Vn1[i-1] + beta[i]*Vn1[i] + gamma[i]*Vn1[i+1] - Vn[i])
    
    #Loss at Last Gridpoint (Backward Differences Only)
    loss.append(
        ((Vn1[-1] - Vn[-1]) / Delta_t
        - 0.5 * sigma**2 * S_N**2 * (Vn1[-1] - 2 * Vn1[-2] + Vn1[-3]) / (Delta_S**2)
        -r * S_N * (Vn1[-1] - Vn1[-2]) / Delta_S
        +r * Vn1[-1])
        )
    
    return np.array(loss)

#------------------------------ Explicit Methods ------------------------------
def explicit_method_noBoundary(Vn,Delta_S,Delta_t):
    
    Vn1 = []
    
    #--- Lower Bound (Forward Differences)
    #Gridpoint
    S_0 = S_grid[0]
    #PDE Loss
    Vn1.append(Vn[0] - Delta_t * (
    0.5 * sigma**2 * S_0**2 * (Vn[2] - 2 * Vn[1] + Vn[0]) / (Delta_S**2)
    + r * S_0 * (Vn[1] - Vn[0]) / Delta_S
    - r * Vn[0])
    )
    
    #--- Interior (Central Differences)
    for i in range(1,len(S_grid)-1):
        #Gridpoint
        S_i = S_grid[i]
        
        #PDE Loss
        Vn1.append(Vn[i] - Delta_t * (
        0.5 * sigma**2 * S_i**2 * (Vn[i+1] - 2 * Vn[i] + Vn[i-1]) / (Delta_S**2)
        + r * S_i * (Vn[i+1] - Vn[i-1]) / (2 * Delta_S)
        - r * Vn[i])
            )
        
    #--- Upper Bound (Backward Differences)
    #Gridpoint
    S_I = S_grid[-1]
    #PDE Loss
    Vn1.append(Vn[-1] - Delta_t * (
        0.5 * sigma**2 * S_I**2 * (Vn[-1] - 2 * Vn[-2] + Vn[-3]) / (Delta_S**2)
        + r * S_I * (Vn[-1] - Vn[-2]) / Delta_S
        - r * Vn[-1])
    )
        
    return np.array(Vn1)

def explicit_method(Vn,Delta_S,Delta_t):
    
    Vn1 = [Vn[0]]
    
    #--- Interior (Central Differences)
    for i in range(1,len(S_grid)-1):
        #Gridpoint
        S_i = S_grid[i]
        
        #PDE Loss
        Vn1.append(Vn[i] - Delta_t * (
        0.5 * sigma**2 * S_i**2 * (Vn[i+1] - 2 * Vn[i] + Vn[i-1]) / (Delta_S**2)
        + r * S_i * (Vn[i+1] - Vn[i-1]) / (2 * Delta_S)
        - r * Vn[i])
            )
        
    Vn1.append(Vn[-1])
        
    return np.array(Vn1)

#%% Solution Option Price Implicit Method

#Loop
Value_Functions_Implicit = [V_terminal]
alpha,beta,gamma,Delta_S,Delta_t = compute_coefficients(S_grid,t_grid) #Always same Coefficients

#------ Solve Value Functions iteratively 
for n in range(0,N_t):
    print("---------------------")
    print("Iteration: " + str(n) + ", t = " + str(t_grid[n+1]))
    
    #Value Function V^n
    Vn  = Value_Functions_Implicit[n]
    
    #Initliase V^{n+1} as V^n
    Vn1 = Vn
    
    #Solve the Root-finding Problem at the collocation points 
    root = scipy.optimize.root(implicit_method, Vn1, args = (Vn,alpha,beta,gamma))
    Vn1 = root.x
    
    #Enforce constraint
    Vn1 = np.maximum(Vn1,payoff(S_grid,K,option_type))
    
    #Store Results
    Value_Functions_Implicit.append(Vn1)
    
    #Print Error
    print("Average L2 Error: " + str(np.linalg.norm(root.fun)/len(root.fun)))
    print("Supremum Error: " + str(np.max(np.abs(root.fun))))
    
#%% Solution Option Price Implicit Method WITHOUT boundary conditions
#Loop
Value_Functions_Implicit_noBoundary = [V_terminal]
alpha,beta,gamma,Delta_S,Delta_t = compute_coefficients(S_grid,t_grid) #Always same Coefficients

#------ Solve Value Functions iteratively 
for n in range(0,N_t):
    print("---------------------")
    print("Iteration: " + str(n) + ", t = " + str(t_grid[n+1]))
    
    #Value Function V^n
    Vn  = Value_Functions_Implicit_noBoundary[n]
    
    #Initliase V^{n+1} as V^n
    Vn1 = Vn
    
    #Solve the Root-finding Problem at the collocation points 
    root = scipy.optimize.root(implicit_method_noBoundary, Vn1, args = (Vn,alpha,beta,gamma,Delta_S,Delta_t))
    Vn1 = root.x
    
    #Enforce constraint
    Vn1 = np.maximum(Vn1,payoff(S_grid,K,option_type))
    
    #Store Results
    Value_Functions_Implicit_noBoundary.append(Vn1)
    
    #Print Error
    print("Average L2 Error: " + str(np.linalg.norm(root.fun)/len(root.fun)))
    print("Supremum Error: " + str(np.max(np.abs(root.fun))))
    
#%% Solution Option Price Explicit Method WITHOUT boundary (cannot get it to work with European Options)
"""
N_t_explicit = int(1e5)
t_grid_explicit = np.linspace(T,0,N_t_explicit+1)
#Loop
Value_Functions_Explicit_noBoundary = [V_terminal]
alpha,beta,gamma,Delta_S,Delta_t_explicit = compute_coefficients(S_grid,t_grid_explicit) #Always same Coefficients

#------ Solve Value Functions iteratively 
for n in range(0,N_t_explicit):
    print("---------------------")
    print("Iteration: " + str(n) + ", t = " + str(t_grid_explicit[n+1]))
    
    #Value Function V^n
    Vn  = Value_Functions_Explicit_noBoundary[n]

    Vn1 = explicit_method_noBoundary(Vn, Delta_S,Delta_t_explicit)
    
    Value_Functions_Explicit_noBoundary.append(Vn1)
"""
    
#%% Solution Option Price Explicit Method with boundary (cannot get it to work with European Options)
"""
N_t_explicit = int(1e5)
t_grid_explicit = np.linspace(T,0,N_t_explicit+1)
#Loop
Value_Functions_Explicit = [V_terminal]
alpha,beta,gamma,Delta_S,Delta_t_explicit = compute_coefficients(S_grid,t_grid_explicit) #Always same Coefficients

#------ Solve Value Functions iteratively 
for n in range(0,N_t_explicit):
    print("---------------------")
    print("Iteration: " + str(n) + ", t = " + str(t_grid_explicit[n+1]))
    
    #Value Function V^n
    Vn  = Value_Functions_Explicit[n]

    Vn1 = explicit_method(Vn, Delta_S,Delta_t_explicit)
    
    Value_Functions_Explicit.append(Vn1)
"""
#%% Comparisons Numerical Solution with Analytical Solution (only Valid for Call Options)

#Analytical Solution Black-Scholes European Call & Put.
def black_scholes(S, K, T, t, r, sigma, option_type):

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    
    if option_type == 'Call':
        return S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    
    if option_type == 'Put':
        return K * np.exp(-r * (T - t)) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    else:
        raise ValueError('Select Call or Put as an Option Type.')



#------ Compare the Implicit Method with analytical solution (comparison only valid for Call Option)
t=0
t_index = int(np.abs(t_grid - t).argmin())

if option_type == 'Call':
    label = 'Analytical'
else:
    label = 'European Analytical'

plt.plot(S_grid,Value_Functions_Implicit[t_index], label='Numerical', linestyle='dashed', linewidth=3)
plt.plot(S_grid,black_scholes(S_grid, K, T, t, r, sigma, option_type), label = label)
plt.legend(fontsize=12)
plt.title(option_type + ", Implicit, t = " + str(t), fontsize=14)
plt.grid(True)
plt.xlabel('Stock Price')
plt.ylabel('Option Price')
plt.savefig(path + "Figures/Comparison_Implicit_" + option_type + ".pdf",dpi = 600, bbox_inches='tight')
plt.show()

#------ Compare the Implicit Method with no Boundaries with analytical solution (comparison only valid for Call Option)

plt.plot(S_grid,Value_Functions_Implicit_noBoundary[t_index], label='Numerical', linestyle='dashed', linewidth=3)
plt.plot(S_grid,black_scholes(S_grid, K, T, t, r, sigma, option_type), label = label)
plt.legend(fontsize=12)
plt.title(option_type + ", Implicit no Boundary Conditions, t = " + str(t), fontsize=14)
plt.grid(True)
plt.xlabel('Stock Price')
plt.ylabel('Option Price')
plt.savefig(path + "Figures/Comparison_Implicit_NoBoundary" + option_type + ".pdf",dpi = 600, bbox_inches='tight')
plt.show()

#%%
def numerical_jacobian(implicit_method, Vn1, Vn, alpha, beta, gamma, epsilon=1e-6):
    """
    Computes the numerical finite difference Jacobian of implicit_method with respect to Vn1.
    """
    n = len(Vn1)  # Size of Vn1
    jacobian = np.zeros((n, n))  # Initialize the Jacobian matrix

    # Evaluate the function at the original Vn1
    f_original = implicit_method(Vn1, Vn, alpha, beta, gamma)

    for j in range(n):
        # Create a perturbed version of Vn1
        Vn1_perturbed = copy.deepcopy(Vn1)
        Vn1_perturbed[j] += epsilon

        # Evaluate the function at the perturbed Vn1
        f_perturbed = implicit_method(Vn1_perturbed, Vn, alpha, beta, gamma)

        # Compute the finite difference approximation for the j-th column
        jacobian[:, j] = (f_perturbed - f_original) / epsilon

    return jacobian

Jac = numerical_jacobian(implicit_method, V_terminal, V_terminal, alpha, beta, gamma, epsilon=1e-6)
