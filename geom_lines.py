
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('my_style.mplstyle')
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter


def par_degrees(x, z, a):
    """
    Calculate the angle in degrees for a given position and parameter 'a' using the par model.
    """
    tan_chi = (1 + a * z**2) / (2 * a * x * z)
    chi = np.arctan(tan_chi)
    chi_deg = np.degrees(chi)
    chi_deg_trans = np.where(chi_deg<0, chi_deg+180, chi_deg) # Transform from (-90,90) to (0,180)
    return chi_deg_trans

def wedge_degrees(x, z, b):
    """
    Calculate the angle in degrees for a given position and parameter 'b' using the wedge model.
    """
    tan_chi = (1 + b * np.abs(z)) / (b * x * np.sign(z))
    chi = np.arctan(tan_chi)
    chi_deg = np.degrees(chi)
    chi_deg_trans = np.where(chi_deg<0, chi_deg+180, chi_deg) # Transform from (-90,90) to (0,180)
    return chi_deg_trans

def const_degrees(x, z, c):
    """
    Calculate the angle in degrees for a given position and parameter 'c' using the const model.
    """
    tan_chi = (np.sign(z) * np.sign(x)) / c
    chi = np.arctan(tan_chi)
    chi_deg = np.degrees(chi)
    chi_deg_trans = np.where(chi_deg<0, chi_deg+180, chi_deg) # Transform from (-90,90) to (0,180)
    return chi_deg_trans

def create_gif(model, lower_bound, upper_bound, name):
    x = np.linspace(-20, 20, 26)
    z = np.linspace(-20, 20, 26)
    xx, zz = np.meshgrid(x, z)

    fig, ax = plt.subplots(figsize=(7, 6))  # Increase figure size for higher resolution
    lim_axes = 20
    ell_maj = 30.
    e = Ellipse((0, 0), width=ell_maj, height=0.1 * ell_maj, angle=0,
                facecolor=(0, 0, 0, 0.3), edgecolor=(0, 0, 0, 1), linestyle='-.')

    param_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)

    def update(frame):
        p = lower_bound + (upper_bound - lower_bound) * frame / 29
        model_degrees = model(xx, zz, p)
        u = np.cos(np.radians(model_degrees))
        w = np.sin(np.radians(model_degrees))
        ax.clear()
        q = ax.quiver(x, z, u, w, scale=30, width=4e-3, color='purple',
                      pivot='middle', headwidth=0, headlength=0, headaxislength=0)
        ax.set_xlim(-lim_axes, lim_axes)
        ax.set_ylim(-lim_axes, lim_axes)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$x\,\mathrm{[kpc]}$")
        ax.set_ylabel(r"$z\,\mathrm{[kpc]}$")
        ax.add_patch(e)
        param_text.set_text(f'Parameter: {p:.2f}')
        ax.add_artist(param_text)
        return q, param_text

    ani = FuncAnimation(fig, update, frames=30, blit=True)
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Michael Stein AIRUB'), bitrate=1800)
    ani.save(f'plots/tests/{name}.mp4', writer=writer, dpi=300)  # Save as MP4 with higher resolution

if __name__ == "__main__":
    for model, p, name in zip([par_degrees, wedge_degrees, const_degrees],
                              [0.01,0.1,1],
                              ['par_test', 'wedge_test', 'const_test']):
        x = np.linspace(-20,20,26)
        z = np.linspace(-20,20,26)
        xx, zz = np.meshgrid(x,z)
        model_degrees = model(xx, zz, p)
        u = np.cos(np.radians(model_degrees))
        w = np.sin(np.radians(model_degrees))        
    
        fig, ax = plt.subplots(figsize=(3.5,3.))
        q = ax.quiver(x, z, u, w, scale=30, width=4e-3, color='purple',
                        pivot='middle', headwidth=0, headlength=0, headaxislength=0)
        lim_axes = 20
        ax.set_xlim(-lim_axes,lim_axes)
        ax.set_ylim(-lim_axes,lim_axes)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$x\,\mathrm{[kpc]}$")
        ax.set_ylabel(r"$z\,\mathrm{[kpc]}$")
        ell_maj=30.
        e = Ellipse((0,0),width=ell_maj, height=0.1*ell_maj, angle=0,
                    facecolor=(0,0,0,0.3), edgecolor=(0,0,0,1), linestyle='-.')
        ax.add_patch(e)  
        plt.savefig(f"plots/tests/{name}.png", bbox_inches='tight', dpi=300)
    
    models = [
        {'model': par_degrees, 'lower_bound': 0.000, 'upper_bound': 0.2, 'name': 'par_test'},
        {'model': wedge_degrees, 'lower_bound': 0.00, 'upper_bound': 1, 'name': 'wedge_test'},
        {'model': const_degrees, 'lower_bound': 0.0, 'upper_bound': 2, 'name': 'const_test'},
        # Add more models and their bounds here
    ]
    for model_info in models:
        create_gif(model_info['model'], model_info['lower_bound'], model_info['upper_bound'], model_info['name'])
