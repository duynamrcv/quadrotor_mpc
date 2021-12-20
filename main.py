import numpy as np
import math
import time
import matplotlib.pyplot as plt

from Quadrotor import Quadrotor
from Plotting import Plotting
from MPCController import AltitudeMPC, AttitudeMPC, PositionMPC

class Trajectory:
    def __init__(self, sim_time=10.0, dt = 0.02):
        self.sim_time = sim_time
        self.dt = dt
        self.ref = self.desiredTrajectory()

        self.x_ref = np.array(self.ref)[:,0]
        self.y_ref = np.array(self.ref)[:,1]
        self.z_ref = np.array(self.ref)[:,2]
        self.psi_ref = np.array(self.ref)[:,3]
    
    def desiredTrajectory(self):
        ref = []
        for i in range(int(self.sim_time/self.dt)):
            t = i*self.dt
            x = math.sin(2*math.pi*t/10)
            y = math.cos(2*math.pi*t/10)
            z = -0.5*t
            yaw = 2*math.pi*t/10
            ref.append([x,y,z,yaw])
        return ref
        
    def desired_altitude(self, quad, idx, N_):
        # initial state / last state
        x_ = np.zeros((N_+1, 2))
        u_ = np.zeros((N_, 1))

        z_ref_ = self.z_ref[idx:(idx+N_)]
        length = len(z_ref_)
        if length < N_:
            z_ex = np.ones(N_ - length)*z_ref_[-1]
            z_ref_ = np.concatenate((z_ref_, z_ex), axis=None)
        
        dz_ref_ = np.diff(z_ref_)
        dz_ref_ = np.concatenate((quad.dpos[2], dz_ref_), axis=None)

        ddz_ref_ = np.diff(dz_ref_)
        ddz_ref_ = np.concatenate((ddz_ref_[0], ddz_ref_), axis=None)

        thrust_ref_ = (quad.g - ddz_ref_)*quad.mq
        
        x_ = np.array([z_ref_, dz_ref_]).T
        x_ = np.concatenate((np.array([[quad.pos[2], quad.dpos[2]]]), x_), axis=0)
        u_ = np.array([thrust_ref_]).T
        # print(x_)
        # print(u_)
        return x_, u_

    def desired_position(self, quad, idx, N_, thrust):
        # initial state / last state
        x_ = np.zeros((N_+1, 4))
        u_ = np.zeros((N_, 2))

        x_ref_ = self.x_ref[idx:(idx+N_)]
        y_ref_ = self.y_ref[idx:(idx+N_)]
        length = len(x_ref_)
        if length < N_:
            x_ex = np.ones(N_ - length)*x_ref_[-1]
            x_ref_ = np.concatenate((x_ref_, x_ex), axis=None)

            y_ex = np.ones(N_ - length)*y_ref_[-1]
            y_ref_ = np.concatenate((y_ref_, y_ex), axis=None)

        dx_ref_ = np.diff(x_ref_)
        dx_ref_ = np.concatenate((quad.dpos[0], dx_ref_), axis=None)
        dy_ref_ = np.diff(y_ref_)
        dy_ref_ = np.concatenate((quad.dpos[1], dy_ref_), axis=None)

        ddx_ref_ = np.diff(dx_ref_)
        ddx_ref_ = np.concatenate((ddx_ref_[0], ddx_ref_), axis=None)
        ddy_ref_ = np.diff(dy_ref_)
        ddy_ref_ = np.concatenate((ddy_ref_[0], ddy_ref_), axis=None)

        the_ref_ = np.arcsin(ddx_ref_*quad.mq/thrust)
        phi_ref_ = -np.arcsin(ddy_ref_*quad.mq/thrust)

        x_ = np.array([x_ref_, y_ref_, dx_ref_, dy_ref_]).T
        x_ = np.concatenate((np.array([[quad.pos[0], quad.pos[1], quad.dpos[0], quad.dpos[1]]]), x_), axis=0)
        u_ = np.array([phi_ref_, the_ref_]).T
        
        # print(x_)
        # print(u_)
        return x_, u_

    def desired_attitude(self, quad, idx, N_, phid, thed):
        # initial state / last state
        x_ = np.zeros((N_+1, 6))
        u_ = np.zeros((N_, 3))

        phi_ref_ = phid
        the_ref_ = thed

        psi_ref_ = self.psi_ref[idx:(idx+N_)]
        length = len(psi_ref_)
        if length < N_:
            psi_ex = np.ones(N_ - length)*psi_ref_[-1]
            psi_ref_ = np.concatenate((psi_ref_, psi_ex), axis=None)

        dphi_ref_ = np.diff(phi_ref_)
        dphi_ref_ = np.concatenate((quad.dori[0], dphi_ref_), axis=None)
        dthe_ref_ = np.diff(the_ref_)
        dthe_ref_ = np.concatenate((quad.dori[1], dthe_ref_), axis=None)
        dpsi_ref_ = np.diff(psi_ref_)
        dpsi_ref_ = np.concatenate((quad.dori[2], dpsi_ref_), axis=None)

        ddphi_ref_ = np.diff(dphi_ref_)
        ddphi_ref_ = np.concatenate((ddphi_ref_[0], ddphi_ref_), axis=None)
        ddthe_ref_ = np.diff(dthe_ref_)
        ddthe_ref_ = np.concatenate((ddthe_ref_[0], ddthe_ref_), axis=None)
        ddpsi_ref_ = np.diff(dpsi_ref_)
        ddpsi_ref_ = np.concatenate((ddpsi_ref_[0], ddpsi_ref_), axis=None)

        tau_phi_ref_ = (quad.Ix*ddphi_ref_ - dthe_ref_*dpsi_ref_*(quad.Iy-quad.Iz))/quad.la
        tau_the_ref_ = (quad.Iy*ddthe_ref_ - dphi_ref_*dpsi_ref_*(quad.Iz-quad.Ix))/quad.la
        tau_psi_ref_ =  quad.Iz*ddpsi_ref_ - dphi_ref_*dthe_ref_*(quad.Ix-quad.Iy)

        x_ = np.array([phi_ref_, the_ref_, psi_ref_, dphi_ref_, dthe_ref_, dpsi_ref_]).T
        x_ = np.concatenate((np.array([[quad.ori[0], quad.ori[1], quad.ori[2], quad.dori[0], quad.dori[1], quad.dori[2]]]), x_), axis=0)
        u_ = np.array([tau_phi_ref_, tau_the_ref_, tau_psi_ref_]).T

        # print(x_)
        # print(u_)
        return x_, u_

# quad = Quadrotor()
# traj = Trajectory()
# traj.desired_altitude(quad, 495, np.array([1,2]), 30)

# exit()

if __name__ == "__main__":
    quad = Quadrotor()

    dt = 0.02
    N = 30
    sim_time = 10.0
    iner = 0

    traj = Trajectory(sim_time, dt)

    al = AltitudeMPC(quad, T=dt, N=N)
    po = PositionMPC(quad, T=dt, N=N)
    at = AttitudeMPC(quad, T=dt, N=N)

    his_thrust = []; his_tau_phi = []; his_tau_the = []; his_tau_psi = []
    his_time = []

    while iner - sim_time/dt < 0.0:
        # print(iner)
        # Solve altitude -> thrust
        next_al_trajectories, next_al_controls = traj.desired_altitude(quad, iner, N)
        thrusts = al.solve(next_al_trajectories, next_al_controls)

        # Solve position -> phid, thed
        next_po_trajectories, next_po_controls = traj.desired_position(quad, iner, N, thrusts)
        phids, theds = po.solve(next_po_trajectories, next_po_controls, thrusts)

        # Solve attitude -> tau_phi, tau_the, tau_psi
        next_at_trajectories, next_at_controls = traj.desired_attitude(quad, iner, N, phids, theds)
        tau_phis, tau_thes, tau_psis = at.solve(next_at_trajectories, next_at_controls)

        quad.updateConfiguration(thrusts[0], tau_phis[0], tau_thes[0], tau_psis[0], dt)
        
        # Store values
        his_thrust.append(thrusts[0])
        his_tau_phi.append(tau_phis[0])
        his_tau_the.append(tau_thes[0])
        his_tau_psi.append(tau_psis[0])
        his_time.append(iner*dt)

        iner += 1
    
    print(np.array(quad.path))

    # Plot Drone
    plot = Plotting("Quadrotor")
    plot.plot_path(quad.path)
    plot.plot_path(traj.ref)

    # Plot control
    plt.figure()
    plt.subplot(221)
    plt.plot(his_time, his_thrust)
    plt.title("The total thrust")
    plt.xlabel("Time [s]")
    plt.ylabel("Value [N]")

    plt.subplot(222)
    plt.plot(his_time, his_tau_phi)
    plt.title("The tau phi")
    plt.xlabel("Time [s]")
    plt.ylabel("Value [N.m]")

    plt.subplot(223)
    plt.plot(his_time, his_tau_the)
    plt.title("The tau theta")
    plt.xlabel("Time [s]")
    plt.ylabel("Value [N.m]")

    plt.subplot(224)
    plt.plot(his_time, his_tau_psi)
    plt.title("The tau psi")
    plt.xlabel("Time [s]")
    plt.ylabel("Value [N.m]")

    plt.show()