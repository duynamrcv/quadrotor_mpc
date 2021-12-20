import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
 
class Plotting:
    def __init__(self, name, xlim=[-1.2,1.2], ylim=[-1.2,1.2], zlim=[0,5], is_grid=True):
        self.fig = plt.figure()
        self.ax = plt.axes(projection ='3d')
        self.ax.set_title(name)
        self.ax.grid(is_grid)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_zlabel('z [m]')
    
    def plot_path(self, path):
        path = np.array(path)
        self.ax.plot(path[:,0], path[:,1], -path[:,2])

    # def plot_ref(self, ref):

# if __name__ == "__main__":
#     path = [[1,2,3,4],
#             [1,0,1,0],
#             [2,3,4,1]]
#     plot = Plotting("Haha")
#     plot.plot_path(path)
#     plt.show()
