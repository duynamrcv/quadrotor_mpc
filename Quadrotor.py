import numpy as np
import math
import time
import matplotlib.pyplot as plt

class Quadrotor:
    def __init__(self, pos=[0,0,0], ori=[0,0,0], dpos=[0,0,0], dori=[0,0,0]):
        # The configuration of quadrotor
        self.pos = np.array(pos)
        self.ori = np.array(ori)
        self.dpos = np.array(dpos)
        self.dori = np.array(dori)

        # The paths
        self.path = [np.append(self.pos, self.ori)]

        # The constant parameters of quadrotor
        self.mq = 1          # Mass of the quadrotor [kg]
        self.g = 9.8         # Gravity [m/s^2]
        self.Ix = 4e-3       # Moment of inertia about Bx axis [kg.m^2]
        self.Iy = 4e-3       # Moment of inertia about By axis [kg.m^2]
        self.Iz = 8.4e-3     # Moment of inertia about Bz axis [kg.m^2]
        self.la = 0.2        # Quadrotor arm length [m]
        self.b = 29e-6       # Thrust coefficient [N.s^2]
        self.d = 1.1e-6      # Drag coefficient [N.m.s^2]

        # The constraints of the quadrotor
        self.max_z = 0
        self.max_phi = 0.5; self.min_phi = -self.max_phi
        self.max_the = 0.5; self.min_the = -self.max_the

        self.max_dx = 1.0; self.min_dx = -self.max_dx
        self.max_dy = 1.0; self.min_dy = -self.max_dy
        self.max_dz = 1.0; self.min_dz = -self.max_dz
        self.max_dphi = math.pi/4; self.min_dphi = -self.max_dphi
        self.max_dthe = math.pi/4; self.min_dthe = -self.max_dthe
        self.max_dpsi = math.pi/4; self.min_dpsi = -self.max_dpsi

        self.max_thrust = 15.0; self.min_thrust = 0.0
        self.max_tau_phi = 5.0; self.min_tau_phi = -self.max_tau_phi
        self.max_tau_the = 5.0; self.min_tau_the = -self.max_tau_the
        self.max_tau_psi = 5.0; self.min_tau_psi = -self.max_tau_psi

    def correctControl(self, thrust, tau_phi, tau_the, tau_psi):
        thrust = min(max(thrust, self.min_thrust), self.max_thrust)
        tau_phi = min(max(tau_phi, self.min_tau_phi), self.max_tau_phi)
        tau_the = min(max(tau_the, self.min_tau_the), self.max_tau_the)
        tau_psi = min(max(tau_psi, self.min_tau_psi), self.max_tau_psi)
        return thrust, tau_phi, tau_the, tau_psi
    
    def correctDotState(self):
        self.dpos[0] = min(max(self.dpos[0], self.min_dx), self.max_dx)
        self.dpos[1] = min(max(self.dpos[1], self.min_dy), self.max_dy)
        self.dpos[2] = min(max(self.dpos[2], self.min_dz), self.max_dz)

        self.dori[0] = min(max(self.dori[0], self.min_dphi), self.max_dphi)
        self.dori[1] = min(max(self.dori[1], self.min_dthe), self.max_dthe)
        self.dori[2] = min(max(self.dori[2], self.min_dpsi), self.max_dpsi)

    def correctState(self):
        self.pos[2] = min(self.pos[2], self.max_z)

        self.ori[0] = min(max(self.ori[0], self.min_phi), self.max_phi)
        self.ori[1] = min(max(self.ori[1], self.min_the), self.max_the)

    def updateConfiguration(self, thrust, tau_phi, tau_the, tau_psi, dt):
        # Prepare the current state
        phi = self.ori[0]
        the = self.ori[1]
        psi = self.ori[2]

        dphi = self.dori[0]
        dthe = self.dori[1]
        dpsi = self.dori[2]

        # The dynamic equations
        thrust, tau_phi, tau_the, tau_psi = self.correctControl(thrust, tau_phi, tau_the, tau_psi)
        # ddx = thrust/self.mq*(np.cos(phi)*np.sin(the)*np.cos(psi) + np.sin(phi)*np.sin(psi))
        # ddy = thrust/self.mq*(np.cos(phi)*np.sin(the)*np.sin(psi) - np.sin(phi)*np.cos(psi))
        # ddz = self.g - thrust/self.mq*(np.cos(phi)*np.cos(the))
        ddx = thrust/self.mq*np.sin(the)
        ddy = -thrust/self.mq*np.sin(phi)
        ddz = self.g - thrust/self.mq
        ddpos = np.array([ddx, ddy, ddz])

        ddphi = (dthe*dpsi*(self.Iy - self.Iz) + tau_phi*self.la)/self.Ix
        ddthe = (dphi*dpsi*(self.Iz - self.Ix) + tau_the*self.la)/self.Iy
        ddpsi = (dphi*dthe*(self.Iz - self.Iy) + tau_psi)/self.Iz
        ddori = np.array([ddphi, ddthe, ddpsi])

        # Update the quadrotor states
        self.dpos = self.dpos + ddpos*dt
        self.dori = self.dori + ddori*dt
        self.correctDotState()

        self.pos = self.pos + self.dpos*dt
        self.ori = self.ori + self.dori*dt
        self.correctState()

        # Add current configuration to paths
        self.path.append(np.append(self.pos, self.ori))

    def updateConfigurationViaSpeed(self, o1, o2, o3, o4, dt):
        # Compute the control vector through angular speed
        thrust = self.b*(o1 + o2 + o3 + o4)
        tau_phi = self.b*(-o2 + o4)
        tau_the = self.b*(o1 - o3)
        tau_psi = self.d*(-o1 + o2 - o3 + o4)

        self.updateConfiguration(thrust, tau_phi, tau_the, tau_psi, dt)