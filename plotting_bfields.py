import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation as R 
#from mayavi import mlab
from aux_functions import create_ellipse
from models import archimedean_spiral, model_c
from model_creation import create_model
from matplotlib.patches import Ellipse


def bfield_plot(x, y, z, u, v, w, rot_y, rot_z,
                x_ext, y_ext, z_ext, apply_mask=True):  

    x_ell, y_ell, z_ell = create_ellipse(rx=12, ry=12, rz=1.5, res_points=15)
    # Rotate to match the position angle of the galaxy:
    r_pos = R.from_euler('Y', rot_y, degrees=True)
    r_incl = R.from_euler('X', rot_z, degrees=True)
    print(f"Rotation matrix to apply position angle:\n{r_pos.as_matrix()}")
    print(f"Rotation matrix to apply inclination angle:\n{r_incl.as_matrix()}")
    r_total = r_pos * r_incl
    print(f"Total rotation matrix:\n{r_total.as_matrix()}")    
    XYZ_ell= np.array([x_ell.ravel(), y_ell.ravel(), z_ell.ravel()]).transpose()
    XYZ_ell_rot = r_total.apply(XYZ_ell)
    x_ellrot = XYZ_ell_rot[:, 0].reshape(x_ell.shape)
    y_ellrot = XYZ_ell_rot[:, 1].reshape(x_ell.shape)
    z_ellrot = XYZ_ell_rot[:, 2].reshape(x_ell.shape)

    # Since the rotation in create_model causes the meshgrid point to exceed the boundaries set in x_ext, y_ext, z_ext,
    # if wanted, the meshgrid can be masked according to x_ext, y_ext, z_ext:
    if apply_mask:
        mask = (np.abs(x) <= x_ext) & (np.abs(y) <= y_ext) & (np.abs(z) <= z_ext)
        x = np.where(mask, x, np.nan)
        y = np.where(mask, y, np.nan)
        z = np.where(mask, z, np.nan)
        u = np.where(mask, u, np.nan)
        v = np.where(mask, v, np.nan)
        w = np.where(mask, w, np.nan)

    mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0))
    mlab.mesh(x_ellrot,y_ellrot,z_ellrot, representation='wireframe',color=(0,0,0), name='Galactic disk indicator')
    b = mlab.quiver3d(x,y,z,u,v,w,
                    scale_mode='none', mode='2darrow', scale_factor=1, name='Magnetic field', opacity=0.5, colormap='plasma')#,vmin=1.e4,vmax=1.e3)
    mlab.colorbar(object=b, title='Magnetic field strength',orientation='vertical',label_fmt='%.1f')
    mlab.outline(extent=[-x_ext, x_ext,-y_ext, y_ext,-z_ext, z_ext])
    mlab.axes()
    mlab.quiver3d(0, -y_ext, 1.5*z_ext, 0, z_ext, 0, color=(0,0,0), name='Line of Sight indicator', mode='arrow', scale_factor=1)
    mlab.xlabel('X in pc')
    mlab.ylabel('Y in pc')
    mlab.zlabel('Z in pc')
    mlab.show()
    
    
def plot_polarization_angles(x, z, pol_ang, field_param,
                            lim_axes=20., ell_ang=0, ell_maj=30, outname=None, scale=50):
    # Convert angles to radians
    angles_rad = np.radians(pol_ang)

    # Compute vector components
    u = np.cos(angles_rad)
    w = np.sin(angles_rad)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=180, clip=False)
    fig, ax = plt.subplots(figsize=(3.5,3.))
    q = ax.quiver(x, z, u, w, pol_ang, cmap='cividis', scale=scale, pivot='middle', norm=norm)
    ax.set_xlim(-lim_axes,lim_axes)
    ax.set_ylim(-lim_axes,lim_axes)
    if len(field_param)>=1:
        ax.set_title(f"Field Parameters: a={field_param[0]}, L={field_param[1]}, B1={field_param[2]}, rot={ell_ang}", fontsize=10)
    
    # Set colorbar
    cbar = plt.colorbar(q, norm=norm)
    cbar.set_label(r'Angle $[^\circ]$')
    e = Ellipse((0,0),width=ell_maj, height=0.1*ell_maj, angle=-ell_ang,
                facecolor=(0,0,0,0.3), edgecolor=(0,0,0,1), linestyle='-.')
    ax.add_patch(e)  
    if outname != None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    # Show the plot
    plt.show()
    
def plot_polarization_angles_lin(x, z, pol_ang, field_param,
                            lim_axes=20., ell_ang=0, ell_maj=30, outname=None, scale=50):
    # Convert angles to radians
    angles_rad = np.radians(pol_ang)

    # Compute vector components
    u = np.cos(angles_rad)
    w = np.sin(angles_rad)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=180, clip=False)
    fig, ax = plt.subplots(figsize=(3.5,3.))
    q = ax.quiver(x, z, u, w, pol_ang, cmap='cividis', scale=scale, pivot='middle', norm=norm)
    ax.set_xlim(-lim_axes,lim_axes)
    ax.set_ylim(-lim_axes,lim_axes)
    if len(field_param)>=1:
        ax.set_title(f"Field Parameters: b={field_param[0]}, L={field_param[1]}, B1={field_param[2]}, rot={ell_ang}", fontsize=10)
    
    # Set colorbar
    cbar = plt.colorbar(q, norm=norm)
    cbar.set_label(r'Angle $[^\circ]$')
    e = Ellipse((0,0),width=ell_maj, height=0.1*ell_maj, angle=-ell_ang,
                facecolor=(0,0,0,0.3), edgecolor=(0,0,0,1), linestyle='-.')
    ax.add_patch(e)  
    if outname != None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    # Show the plot
    plt.show()    
    
def plot_polarization_angles_check_rot(x1, z1, pol_ang1,
                                       x2, z2, pol_ang2,
                                       lim_axes=20.,scale=50, outname=None):
    # Convert angles to radians
    norm = matplotlib.colors.Normalize(vmin=0, vmax=180, clip=False)
    angles_rad1 = np.radians(pol_ang1)
    # Compute vector components
    u1 = np.cos(angles_rad1)
    w1 = np.sin(angles_rad1)
    angles_rad2 = np.radians(pol_ang2)
    # Compute vector components
    u2 = np.cos(angles_rad2)
    w2 = np.sin(angles_rad2)
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
    q1 = ax1.quiver(x1, z1, u1, w1, pol_ang1, cmap='cividis', scale=scale, pivot='middle', norm=norm)
    ax1.set_xlim(-lim_axes,lim_axes)
    ax1.set_ylim(-lim_axes,lim_axes)
    ax1.set_xlabel('[kpc]')
    ax1.set_ylabel('[kpc]')
    cbar1 = plt.colorbar(q1, norm=norm)
    cbar1.set_label(r'Angle $[^\circ]$')
    ax1.set_title("Original")
    q2 = ax2.quiver(x2, z2, u2, w2, pol_ang2, cmap='cividis', scale=scale, pivot='middle', norm=norm)
    ax2.set_xlim(-lim_axes,lim_axes)
    ax2.set_ylim(-lim_axes,lim_axes)
    ax2.set_xlabel(' X [kpc]')
    ax2.set_ylabel('Z [kpc]')
    ax2.set_title("Rotated")
    ax2.set_xlabel('Galaxy Major Axis (X) [kpc]')
    ax2.set_ylabel('Galaxy Minor Axis (Z) [kpc]')
    cbar2 = plt.colorbar(q2, norm=norm)
    cbar2.set_label(r'Angle $[^\circ]$')
    if outname != None:
        fig.savefig(outname, bbox_inches='tight', dpi=300)

def plot_polarization_vectors(x, z, u, w, field_param, norm_vectors=True,
                              lim_axes=20., ell_ang=0, ell_maj=30, outname=None, scale=50):
    # Normalize vectors for consistent length
    if norm_vectors:
        length = 1.0
        vec_norm = np.sqrt(u**2 + w**2)
        u = length * u / vec_norm
        w = length * w / vec_norm
        print(np.mean(np.sqrt(u**2 + w**2)))
        
    model_ang = np.degrees(np.arctan2(w, u))
    print(f"Model ang min: {np.amin(model_ang)}, max: {np.amax(model_ang)}")
    model_ang = np.where(model_ang < 0, model_ang+180, model_ang)
    #model_ang = np.where(model_ang > 90, model_ang-180, model_ang)
    # Plot arrows with specified positions
    fig, ax = plt.subplots(figsize=(3.5,3.))
    q = ax.quiver(x, z, u, w, model_ang, cmap='cividis', scale=scale, pivot='middle')
    ax.set_xlim(-lim_axes,lim_axes)
    ax.set_ylim(-lim_axes,lim_axes)
    if len(field_param)>=1:
        ax.set_title(f"Field Parameters: a={field_param[0]}, L={field_param[1]}, B1={field_param[2]}", fontsize=10)
    
    # Set colorbar
    cbar = plt.colorbar(q, ticks=np.arange(0, 181, 30))
    cbar.ax.set_yticklabels([str(i) for i in np.arange(0, 181, 30)])
    cbar.set_label(r'Angle $[^\circ]$')
    e = Ellipse((0,0),width=ell_maj, height=0.1*ell_maj, angle=-ell_ang,
                facecolor=(0,0,0,0.3), edgecolor=(0,0,0,1), linestyle='-.')
    ax.add_patch(e)  
    if outname != None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    # Show the plot
    plt.show()  
    
def plot_polarization_vectors_lin(x, z, u, w, field_param, norm_vectors=True,
                              lim_axes=20., ell_ang=0, ell_maj=30, outname=None, scale=50):
    # Normalize vectors for consistent length
    if norm_vectors:
        length = 1.0
        vec_norm = np.sqrt(u**2 + w**2)
        u = length * u / vec_norm
        w = length * w / vec_norm
        print(np.mean(np.sqrt(u**2 + w**2)))
        
    model_ang = np.degrees(np.arctan2(w, u))
    model_ang = np.where(model_ang < 0, model_ang+180, model_ang)
    #model_ang = np.where(model_ang > 90, model_ang-180, model_ang)
    # Plot arrows with specified positions
    fig, ax = plt.subplots(figsize=(3.5,3.))
    q = ax.quiver(x, z, u, w, model_ang, cmap='cividis', scale=scale, pivot='middle')
    ax.set_xlim(-lim_axes,lim_axes)
    ax.set_ylim(-lim_axes,lim_axes)
    if len(field_param)>=1:
        ax.set_title(f"Field Parameters: b={field_param[0]}, L={field_param[1]}, B1={field_param[2]}", fontsize=10)
    
    # Set colorbar
    cbar = plt.colorbar(q, ticks=np.arange(0, 181, 30))
    cbar.ax.set_yticklabels([str(i) for i in np.arange(0, 181, 30)])
    cbar.set_label(r'Angle $[^\circ]$')
    e = Ellipse((0,0),width=ell_maj, height=0.1*ell_maj, angle=-ell_ang,
                facecolor=(0,0,0,0.3), edgecolor=(0,0,0,1), linestyle='-.')
    ax.add_patch(e)  
    if outname != None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    # Show the plot
    plt.show()      
    
    
def fit_result_plot(data, model, outname, opt_params):
    residual = data - model
    abs_residual = np.abs(residual)
    med = np.nanmedian(abs_residual.ravel())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    d = ax1.imshow(data, cmap='cividis', vmin=0, vmax=180)
    ax1.set_title('Data')
    m = ax2.imshow(model, cmap='cividis', vmin=0, vmax=180)
    ax2.set_title('Model')
    r = ax3.imshow(data-model, cmap='PRGn')  
    ax3.set_title('Residual')
    ax3.set_facecolor('grey')
    ax4.hist(abs_residual.ravel(), bins=int(180./5.))
    ax4.axvline(med, color='black', linestyle='dashed')
    min_ylim, max_ylim = ax4.get_ylim()
    ax4.text(med + 0.1 *np.nanmax(residual) , max_ylim*0.9,
             f"Median: {med:.1f}")
    ax4.set_title('Absolute Residual')
    ax4.text(0.3 * np.nanmax(residual), max_ylim*0.5,
                f"a={opt_params[0]:.2f}; rot={opt_params[1]:.1f}")
    plt.colorbar(d, ax=ax1)
    plt.colorbar(m, ax=ax2)
    plt.colorbar(r, ax=ax3)
    fig.tight_layout()
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()
    
def fit_result_plot_II(x, z, data, model, title, outname, lim_axes=10):
    residual = data - model
    abs_residual = np.abs(residual)
    med = np.nanmedian(abs_residual.ravel())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    d = ax1.scatter(x, z, c=data, cmap='inferno', s=7, vmin=0, vmax=180)
    ax1.set_title('Data')
    ax1.set_xlim(-lim_axes,lim_axes)
    ax1.set_ylim(-lim_axes,lim_axes)
    ax1.set_xlabel('Major Axis [kpc]')
    ax1.set_ylabel('Minor Axis [kpc]')
    m = ax2.scatter(x, z, c=model, cmap='inferno', s=7, vmin=0, vmax=180)
    ax2.set_title('Model')
    ax2.set_xlim(-lim_axes,lim_axes)
    ax2.set_ylim(-lim_axes,lim_axes)
    ax2.set_xlabel('Major Axis [kpc]')
    ax2.set_ylabel('Minor Axis [kpc]')
    r = ax3.scatter(x, z, c=data-model, cmap='PRGn', s=7)  
    ax3.set_title('Residual')
    ax3.set_xlim(-lim_axes,lim_axes)
    ax3.set_ylim(-lim_axes,lim_axes)
    ax3.set_xlabel('Major Axis [kpc]')
    ax3.set_ylabel('Minor Axis [kpc]')
    ax3.set_facecolor('grey')
    ax4.hist(abs_residual.ravel(), color='grey')
    ax4.axvline(med, color='black', linestyle='dashed')
    ax4.set_xlabel(r'Angular Deviation $[^\circ]$')
    ax4.set_ylabel(r"$N_{\mathrm{pix}}$")
    min_ylim, max_ylim = ax4.get_ylim()
    ax4.text(med + 0.1 *np.nanmax(residual) , max_ylim*0.8,
             f"Median: {med:.1f}")
    ax4.set_title('Absolute Residual')
    cbar1 = plt.colorbar(d, ax=ax1)
    cbar2 = plt.colorbar(m, ax=ax2)
    cbar3 = plt.colorbar(r, ax=ax3)
    cbar1.set_label(r'Angle $[^\circ]$')
    cbar2.set_label(r'Angle $[^\circ]$')
    cbar3.set_label(r'Angle $[^\circ]$')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()
    
    

    
if __name__=='__main__':
    bfield_plot(field_function=archimedean_spiral, function_parameters=[1,0.1], position_angle=20, inclination_angle=3, n_x=20,n_y=20, n_z=20)
    x, y, z, u, v, w = create_model(field_function=model_c, function_parameters=[0.01,5,30],
                                    rot_y=45, rot_z=10, n_x=25,n_y=25, n_z=25, x_size=20, y_size=20, z_size=20)
    bfield_plot(x, y, z, u, v, w, rot_y=45, rot_z=10, x_ext=20, y_ext=20, z_ext=20)
    x, y, z, u, v, w = create_model(field_function=model_c, function_parameters=[0.5,5,30],
                                    rot_y=45, rot_z=10, n_x=25,n_y=1, n_z=25, x_size=20, y_size=2, z_size=20)
    bfield_plot(x, y, z, u, v, w, rot_y=45, rot_z=10, x_ext=20, y_ext=20, z_ext=20)