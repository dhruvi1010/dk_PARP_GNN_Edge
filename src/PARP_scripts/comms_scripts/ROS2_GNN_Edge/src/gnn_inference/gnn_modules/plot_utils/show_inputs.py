# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : ploting functions
# ---------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
def plot_measurements(
    x, y, size, figsize = (8, 6),
    xlim_min=-10, xlim_max=10, ylim_min=-5, ylim_max=5,save_path=None):
    
    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, size, color='red', marker='o', label='Measurements')
    ax.set_xlabel('X (Meters)', fontsize=14)
    ax.set_ylabel('Y (Meters)', fontsize=14)
    ax.legend(loc='upper right', fontsize='large')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    ax.tick_params(axis='both', labelsize=14)
    plt.title('Raw Radar Point Cloud', fontsize=14)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')
        print(f"✅ Plot saved at: {save_path}")
    plt.show()

# ---------------------------------------------------------------------------------------------------------------
def plot_graph(
    meas_px, meas_py, edge_coordinates, figsize=(8,6),
    xlim_min=-10, xlim_max=10, ylim_min=-5, ylim_max=5,save_path=None):

    nodes_x = np.stack((meas_px[edge_coordinates[0]], meas_px[edge_coordinates[1]]), axis=-1)
    nodes_y = np.stack((meas_py[edge_coordinates[0]], meas_py[edge_coordinates[1]]), axis=-1)
    _, ax = plt.subplots(figsize=figsize)
    ax.plot(nodes_x.T, nodes_y.T, color='k', marker='.', markersize=1, markeredgecolor='none', linewidth=0.75)
    ax.scatter(meas_px, meas_py, 30, color='red', marker='o') #size 30
    ax.set_xlabel('X (Meters)', fontsize=14)
    ax.set_ylabel('Y (Meters)', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max)
    ax.tick_params(axis='both', labelsize=14)
    plt.title('Generated Graph with Nodes and Edges', fontsize=14)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')
        print(f"✅ Plot saved at: {save_path}")
    plt.show()