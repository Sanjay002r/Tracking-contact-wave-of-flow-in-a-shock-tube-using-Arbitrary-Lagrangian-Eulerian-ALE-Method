###################################################################################################################
#             Tracking contact wave with moving grid formulation using ALE for 1D shock tube problem              #
#                                               AUTHOR: Sanjay R                                                  #
#                                                                                                                 #
###################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

# Constants and Initial Conditions
Imax = 51  # Number of grid points in 1D
Lx = 1.0  # Length of the domain
R = 287.04  # Gas constant
gamma = 1.4  # Specific heat ratio
CFL = 0.8  # CFL number

# 1D grid
x = np.linspace(0, Lx, Imax)
w = np.zeros(Imax)
xc=(x[:-1] + x[1:])/2 #Cell centroids

# Combined function for predictor, corrector, and spring analogy
def dynamic_mesh(delta_x_prev, delta_x_prev_prev, x, Imax, w_middle, dt, max_iterations=1000, tolerance=1e-9):
    # Move central grid point first
    delta_x_new = np.copy(delta_x_prev)
    delta_x_new[Imax//2] = w_middle * dt

    # Predictor step: Linear extrapolation
    delta_x_new[:Imax//2:] = 2 * delta_x_new[:Imax//2:] - delta_x_prev_prev[:Imax//2:]

    # Corrector Jacobi iterations
    for _ in range(max_iterations):
        delta_x_iter = np.copy(delta_x_new)

        # Corrector using spring analogy
        for i in range(1, Imax - 1):  # Ignore boundary nodes
            if i==Imax//2:
                delta_x_new[i]+= 0
            else: 
                k_sum_x = 0
                weighted_sum_x = 0
                
                # Neighbors are i-1 and i+1 for a 1D grid
                for j in [i - 1, i + 1]:
                    km = 1 / abs(x[i] - x[j]) if x[i] != x[j] else 0  # Spring stiffness, 0 if accidentally nodes overlap
                    weighted_sum_x += km * delta_x_iter[j]
                    k_sum_x += km

                # Update delta_x_new for node i
                delta_x_new[i] = weighted_sum_x / k_sum_x if k_sum_x != 0 else delta_x_iter[i]

        # Check for convergence
        if np.linalg.norm(delta_x_new - delta_x_iter) < tolerance:
            break
    
    return delta_x_new, x + delta_x_new

def conservative():
    # Initial conditions: High pressure on the left, low on the right
    p = np.ones(Imax-1)
    p[:(Imax-1)//2] = 101325 *5  # High pressure in the left half
    p[(Imax-1)//2:] = 101325  # Low pressure in the right half
    T = np.ones(Imax-1) * 300
    rho = p / (R * T)  # Density from the state equation
    u = np.zeros(Imax-1)  # Initial velocity

    # Compute energy per unit volume
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return np.array([rho, rho * u, E])

def primitive_vars(U):
    # Extract primitive variables from conservative variables
    rho = U[0]
    u = U[1] / rho
    E = U[2]
    p=(gamma-1)*(E-0.5*rho*(u**2))
    H=(E+p)/rho
    T=p/(287.04*rho)
    a=np.sqrt(gamma*287.04*T)
    M=np.sqrt(u**2)/a
    return rho,u,E,p,T,H,a,M

# Geometric Conservation Law function to compute new dx (cell lengths) 
# Not used here as results were uncontrollable, but need to check later
def GCL(x, w, dt):
    dx = x[1:] - x[:-1]
    dx_new = dx + dt * (w[1:] - w[:-1])
    return dx_new

# Compute the flux for 1D Roe scheme
def Roe_Scheme_ALE(rho, p, u, E, H, w, U, xnew, face):
    # Compute primitive variables and related quantities
    tilda_rho = np.sqrt(rho[:-1] * rho[1:])
    temp = np.sqrt(rho[:-1]) / (np.sqrt(rho[:-1]) + np.sqrt(rho[1:]))
    tilda_u = (temp * u[:-1] + (1-temp) * u[1:])
    tilda_H = (temp * H[:-1] + (1-temp) * H[1:])
    delta_p = p[1:] - p[:-1]
    delta_rho = rho[1:] - rho[:-1]
    delta_u = u[1:] - u[:-1]
    tilda_a = np.sqrt((gamma - 1) * (tilda_H - 0.5 * tilda_u**2))

    # Eigenvectors for 1D (K1, K2, K3)
    K1 = np.array([np.ones_like(tilda_a), tilda_u + tilda_a, tilda_H + tilda_a * tilda_u])
    K2 = np.array([np.ones_like(tilda_a), tilda_u, 0.5 * tilda_u**2])
    K3 = np.array([np.ones_like(tilda_a), tilda_u - tilda_a, tilda_H - tilda_a * tilda_u])
    
    # Wave strengths
    alpha_1 = (delta_p + tilda_rho * tilda_a * delta_u) / (2 * tilda_a ** 2)
    alpha_2 = delta_rho - (delta_p / tilda_a ** 2)
    alpha_3 = (delta_p - tilda_rho * tilda_a * delta_u) / (2 * tilda_a ** 2)

        # Compute weighted interpolation for w
    dx = xnew[1:] - xnew[:-1]  # Distances between points in xnew
    weights = dx / dx.sum()    # Normalize to get weights

    # Apply weighted interpolation using weights from xnew
    w_interp = (weights * w[1:] + (1 - weights) * w[:-1])

    F_1D = np.array([rho*(u-w_interp), (rho*(u-w_interp)*u + p), E*(u-w_interp)+p*u])
    
    F_face = np.zeros((3, Imax - 1))

    if face==1:
          # Eigenvalues (1D case: u - a, u, u + a)
        lambda_1 = np.abs(tilda_u + tilda_a - w[:-2])
        lambda_2 = np.abs(tilda_u - w[:-2])
        lambda_3 = np.abs(tilda_u - tilda_a - w[:-2])

        for i in range(Imax - 2):
            F_face[:, i] = 0.5 * (F_1D[:, i] - w[i] * U[:,i] + F_1D[:, i + 1]- w[i] * U[:,i+1] ) - 0.5 * (alpha_1[i] * K1[:, i] * lambda_1[i] +alpha_2[i] * K2[:, i] * lambda_2[i] + alpha_3[i] * K3[:, i] * lambda_3[i])
    else:    
        # Eigenvalues (1D case: u - a, u, u + a)
        lambda_1 = np.abs(tilda_u + tilda_a - w[1:-1])
        lambda_2 = np.abs(tilda_u - w[1:-1])
        lambda_3 = np.abs(tilda_u - tilda_a - w[1:-1])

        for i in range(Imax - 2):
            F_face[:, i+1] = 0.5 * (F_1D[:, i] - w[i+1] * U[:,i] + F_1D[:, i + 1]- w[i+1] * U[:,i+1]) - 0.5 * (alpha_1[i] * K1[:, i] * lambda_1[i] + alpha_2[i] * K2[:, i] * lambda_2[i] + alpha_3[i] * K3[:, i] * lambda_3[i])

    return F_face


# Function to solve the 1D system
def solve(U=conservative()):
    t = 0
    Unew = U.copy()
    RSS = 1.0
    history = np.array([[t, RSS]])
    x_history = np.array([[x]])
    dx = x[1:] - x[:-1]
    delta_x_prev = np.zeros(Imax)
    delta_x_prev_prev = np.zeros(Imax)
    
    while t<=0.75e-3:
        U = Unew.copy()
        rho, u, E, p, T, H, a, M = primitive_vars(U)  # Extract variables from U

        w_middle = 0* u[Imax//2] #1* 0.5*(u[Imax//2]+u[Imax//2+1])#  # Velocity at the middle grid point to move at next time step       
        #print(w_middle)
        # Time step calculation (using 1D fluxes)
        lambda_x = (np.abs(u) + a)
        dt = np.min(CFL * dx / lambda_x)
        print(w_middle*dt)
        # Update displacements and newer grid with dynamic mesh
        delta_x_new, x_new = dynamic_mesh(delta_x_prev, delta_x_prev_prev, x_history[-1,0], Imax, w_middle, dt)
        w = (x_new - x_history[-1,0,:])/dt
        dx_new= x_new[1:]- x_new[:-1]

        # dx_new = GCL(x_new, delta_x_new, dt=dt) # Calculate new dx values using GCL
        # w = (x_new-x)/dt

        F1 = Roe_Scheme_ALE(rho, p, u, E, H, w, U, x_new, face=1)  # Flux at face 1
        F2 = Roe_Scheme_ALE(rho, p, u, E, H, w, U, x_new, face=2)  # Flux at face 2 

        # Updating Unew for all interior points
        for i in range(1, Imax-2):
                Unew[:, i] = U[:, i]*dx[i]/dx_new[i] - dt/dx_new[i] * (F1[:, i] - F2[:, i])
        Unew[:, 0] = Unew[:, 1]  # Left boundary zero-gradient
        Unew[:, -1] = Unew[:, -2]    # Right boundary zero-gradient

        delta_x_prev_prev, delta_x_prev = delta_x_prev.copy(), delta_x_new.copy()
        dx=dx_new.copy()
        
        t += dt

        RSS = np.sqrt(np.sum(((Unew[0] - U[0]) / U[0]) ** 2) / (Imax)) # Calculate RSS of density and check convergence
        print("Time step and RSS value",history[-1])
        history = np.append(history, np.array([[t, RSS]]), axis=0)
        x_history= np.append(x_history, np.array([[x_new]]), axis=0)
        print("Final t",t)
        
    return U, np.array(history), x_history


def final_plotter(p, M, u, E, T, rho, history,x_history):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    xc = 0.5*(x_history[-1,0,1:]+x_history[-1,0,:-1])
    # Pressure Line Plot
    axs[0, 0].plot(xc, p, color='blue')
    axs[0, 0].set_title('Pressure Line Plot')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Pressure')
    axs[0, 0].grid()

    # Mach Line Plot
    axs[0, 1].plot(xc, M, color='green')
    axs[0, 1].set_title('Mach Number Line Plot')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Mach Number')
    axs[0, 1].grid()

    # Normalized Pressure Variations
    axs[1, 0].plot(xc, p / np.max(p), label='Normalized Pressure', color='purple')
    axs[1, 0].set_title('Normalized Pressure Variations')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Normalized Pressure')
    axs[1, 0].grid()
    axs[1, 0].legend()

    # Plot Convergence Plot
    axs[1, 1].plot(np.arange(len(history[:, 1])), history[:, 1])
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Convergence Plot')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('RSS')
    axs[1, 1].grid()
    plt.tight_layout()  # Adjust layout to take full space
    plt.show()

    plt.figure()
    cmap = plt.cm.viridis  # You can choose any other colormap like 'plasma', 'inferno', etc.
    for i in range(len(history[:, 1])):
        if i % 5 == 0:
            color = cmap(i / len(history[:, 1]))  # Normalize i to map to the colormap range
            plt.plot(x_history[i, 0, :], np.full_like(x_history[i, 0, :], i), 'o', color=color, markersize=4, label=f"Time = {history[i, 0]:.2e}")
            plt.plot(x_history[i, 0, Imax // 2], i, 'ro', markersize=8)  # Marking mid-point

    plt.title("1D Grid Point Movements Over Time")
    plt.xlabel("Grid Points Position")
    plt.ylabel("Iterations")
    plt.grid()
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # Pressure Line Plot
    axs[0, 0].plot(xc, rho, color='blue')
    axs[0, 0].set_title('Density Line Plot')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].grid()

    # Mach Line Plot
    axs[0, 1].plot(xc, E, color='orange')
    axs[0, 1].set_title('Energy Line Plot')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Energy')
    axs[0, 1].grid()

    # Mach Line Plot
    axs[1, 0].plot(xc, u, color='black')
    axs[1, 0].set_title('Velocity Line Plot')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Velocity')
    axs[1, 0].grid()

    # Mach Line Plot
    axs[1, 1].plot(xc, T, color='green')
    axs[1, 1].set_title('Temperature Line Plot')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Temperature')
    axs[1, 1].grid()

    plt.tight_layout()  # Adjust layout to take full space
    plt.show()

if __name__ == "__main__":
    # plot_mesh(X,Y) # To view the mesh in which problem is solved
    U, history, x_history = solve() # Solve the problem
    print(x_history.shape)
    rho,u,E,p,T,H,a,M = primitive_vars(U) # Extract primitive variables from the solution
    final_plotter(p, M,u,E, T, rho, history, x_history) # Plot the final results in cell centroids (without interpolation, where still the phenomenon is captured perfectly with the plots)
