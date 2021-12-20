import casadi as ca
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# the robot params
mq = 1          # Mass of the quadrotor [kg]
g = 9.8         # Gravity [m/s^2]
Ix = 4e-3       # Moment of inertia about Bx axis [kg.m^2]
Iy = 4e-3       # Moment of inertia about By axis [kg.m^2]
Iz = 8.4e-3     # Moment of inertia about Bz axis [kg.m^2]
la = 0.2        # Quadrotor arm length [m]
b = 29e-6       # Thrust coefficient [N.s^2]
d = 1.1e-6      # Drag coefficient [N.m.s^2]

def shift(T, t0, x0, u, x_n, f):
    f_value = f(x0, u[0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return t, st, u_end, x_n

def predict_state(x0, u, T, N):
    states = np.zeros((N+1, 2))
    states[0,:] = x0
    # euler method
    for i in range(N):
        states[i+1, 0] = states[i, 1]*T

        states[i+1, 1] = (g - u[i, 1]/mq)*T
    return states

def desired_command_and_trajectory(t, T, x0_:np.array, N_):
    # initial state / last state
    x_ = np.zeros((N_+1, 2))
    x_[0] = x0_
    u_ = np.zeros((N_, 1))
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + T*i
        z_ref_ = -2.0
        dz_ref_ = 0
        ddz_ref_ = 0

        thrust_ref_ = (g - ddz_ref_)*mq

        x_[i+1] = np.array([z_ref_, dz_ref_])
        u_[i] = np.array([thrust_ref_])
    # return pose and command
    return x_, u_

if __name__ == "__main__":
    # mpc parameters
    T = 0.02    # time step
    N = 30      # predict hoziron length

    opti = ca.Opti()
    # the total thrust and torques of all axis
    opt_controls = opti.variable(N, 1)
    thrust = opt_controls

    # state variable: altitude position
    opt_states = opti.variable(N+1, 2)
    z = opt_states[:,0]
    dz = opt_states[:,1]

    # create model
    f = lambda x_, u_: ca.vertcat(*[
        x_[1],
        g - u_/mq,
    ])
    f_np = lambda x_, u_: np.array([
        x_[1],
        g - u_/mq,
    ])

    # parameters, these parameters are the reference trajectories of the pose and inputs
    opt_u_ref = opti.parameter(N, 1)
    opt_x_ref = opti.parameter(N+1, 2)

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x_ref[0, :])
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)
    
    # weight matrix
    Q = np.diag([30.0, 10.0])
    R = np.diag([1.0])

    # cost function
    obj = 0
    for i in range(N):
        state_error_ = opt_states[i, :] - opt_x_ref[i+1, :]
        control_error_ = opt_controls[i, :] - opt_u_ref[i, :]
        obj = obj + ca.mtimes([state_error_, Q, state_error_.T]) + ca.mtimes([control_error_, R, control_error_.T])
    opti.minimize(obj)

    # boundary and control conditions
    opti.subject_to(opti.bounded(-math.inf, z, 0))
    opti.subject_to(opti.bounded(-1.0, dz, 1.0))

    opti.subject_to(opti.bounded(-0, thrust, 12))

    opts_setting = {'ipopt.max_iter':2000,
                    'ipopt.print_level':0,
                    'print_time':0,
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6}

    opti.solver('ipopt', opts_setting)

    #######
    t0 = 0
    init_state = np.array([0.0, 0.0])
    current_state = init_state.copy()
    u0 = np.zeros((N, 1))
    next_trajectories = np.tile(init_state, N+1).reshape(N+1, -1) # set the initial state as the first trajectories for the robot
    next_controls = np.zeros((N, 1))
    next_states = np.zeros((N+1, 2))
    x_c = [] # contains for the history of the state
    u_c = [u0[0]]
    t_c = [t0] # for the time
    xx = [current_state]
    xr = [next_trajectories[0]]
    ur = [next_controls[0]]
    sim_time = 10.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(mpciter-sim_time/T<0.0):
        ## set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x_ref, next_trajectories)
        opti.set_value(opt_u_ref, next_controls)
        ## provide the initial guess of the optimization targets
        opti.set_initial(opt_controls, u0.reshape(N, 1))# (N, 3)
        opti.set_initial(opt_states, next_states) # (N+1, 6)
        ## solve the problem once again
        t_ = time.time()
        sol = opti.solve()
        index_t.append(time.time()- t_)
        ## obtain the control input
        u_res = sol.value(opt_controls)
        x_m = sol.value(opt_states)
        # print(x_m[:3])
        u_c.append(u_res[0])
        t_c.append(t0)
        x_c.append(x_m)
        t0, current_state, u0, next_states = shift(T, t0, current_state, u_res, x_m, f_np)
        xx.append(current_state)
        ## estimate the new desired trajectories and controls
        next_trajectories, next_controls = desired_command_and_trajectory(t0, T, current_state, N)
        xr.append(next_trajectories[1])
        ur.append(next_controls[0])
        mpciter = mpciter + 1
    
    ## after loop
    print(mpciter)
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))

    # Plot Attitude Tracking
    plt.figure()
    plt.plot(t_c, -np.array(xr)[:,0], label='ref z')
    plt.plot(t_c, -np.array(xx)[:,0], label='quad z')
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("value [m]")
    plt.grid(True)
    plt.title("Altitude tracking")

    # Plot Thrust force
    plt.figure()
    plt.step(t_c, np.array(u_c), label='quad thrust')
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("value [N]")
    plt.grid(True)
    plt.title("Thrust force")

    plt.show()