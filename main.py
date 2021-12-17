import numpy as np
import math
import time
import matplotlib.pyplot as plt

from Quadrotor import Quadrotor
from Plotting import Plotting

if __name__ == "__main__":
    quad = Quadrotor()
    thrust = 12
    tau_phi = 0.01; tau_the = 0.01; tau_psi = 0.01

    dt = 0.1
    sim_time = 5
    iner = 0
    while iner - sim_time/dt < 0.0:
        print(iner)
        quad.updateConfiguration(thrust, tau_phi, tau_the, tau_psi, dt)
        iner += 1
    
    print(np.array(quad.path))
    plot = Plotting("Quadrotor")
    plot.plot_path(quad.path)
    plt.show()