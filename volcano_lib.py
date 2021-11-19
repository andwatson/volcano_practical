#!/usr/bin/env python3
'''
## volcano_lib.py

Library of python functions to be used with volcano_practical.ipynb.

'''

# packages
import subprocess as subp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmcrameri import cm
import warnings

#-------------------------------------------------------------------------------

def mogi_source(x,y,xcen=0,ycen=0,d=3e3,dV=1e6, nu=0.25):
    
    '''
    Based on the Mogi source provided by Scott Henderson at:
    https://github.com/scottyhq/cov9/blob/master/mogi.py
    
    Original Mogi source from:
    Mogi 1958, Segall 2010 p.203
    
    INPUTS:
        x = x-coordinate grid (m)
        y = y-coordinate grid (m)
        
        xcen = y-offset of point source epicenter (m) (default = 0)
        ycen = y-offset of point source epicenter (m) (default = 0)
        d = depth to point (m) (default = 3e3)
        dV = change in volume (m^3) (default = 1e6)
        nu = poisson's ratio for medium (default = 0.25)
        
    OUTPUTS:
        ux = displacement in x-direction at each point in (x,y)
        uy = displacement in y-direction at each point in (x,y)
        uz = displacement in z-direction at each point in (x,y)
    
    '''
    
    # Centre coordinate grid on point source
    x = x - xcen
    y = y - ycen
    
    # Convert to surface cylindrical coordinates
    th, rho = cart2pol(x,y)
    R = np.hypot(d,rho)
    
    # Mogi displacement calculation
    C = ((1-nu) / np.pi) * dV
    ur = C * rho / R**3
    uz = C * d / R**3
    
    # Convert back to cartesian coordinates
    ux, uy = pol2cart(th, ur)
    
    # reshape to input grid size
    ux = ux.reshape(x.shape)
    uy = uy.reshape(x.shape)
    uz = uz.reshape(x.shape)
    
    return ux, uy, uz

#-------------------------------------------------------------------------------

def mogi_source_topo(x, y, elev, xcen=0, ycen=0, d=3e3, dV=1e6, nu=0.25):
    
    '''
    Based on the Mogi source provided by Scott Henderson at:
    https://github.com/scottyhq/cov9/blob/master/mogi.py
    
    Includes an adjustment for topography above the source.
    Elevations are measured above sea level, source depth is measured below sea level.
    
    Original Mogi source from:
    Mogi 1958, Segall 2010 p.203
    
    INPUTS:
        x = x-coordinate grid (m)
        y = y-coordinate grid (m)
        
        xcen = y-offset of point source epicenter (m) (default = 0)
        ycen = y-offset of point source epicenter (m) (default = 0)
        d = depth to point (m) (default = 3e3)
        dV = change in volume (m^3) (default = 1e6)
        nu = poisson's ratio for medium (default = 0.25)
        
        elev = array of elevations, same size as x and y (m)
        
    OUTPUTS:
        ux = displacement in x-direction at each point in (x,y)
        uy = displacement in y-direction at each point in (x,y)
        uz = displacement in z-direction at each point in (x,y)
    
    '''
    
    # Centre coordinate grid on point source
    x = x - xcen
    y = y - ycen
    
    # Convert to surface cylindrical coordinates
    th, rho = cart2pol(x,y)
    R = np.hypot(d+elev,rho)
    
    # Mogi displacement calculation
    C = ((1-nu) / np.pi) * dV
    ur = C * rho / R**3
    uz = C * d / R**3
    
    # Convert back to cartesian coordinates
    ux, uy = pol2cart(th, ur)
    
    # reshape to input grid size
    ux = ux.reshape(x.shape)
    uy = uy.reshape(x.shape)
    uz = uz.reshape(x.shape)
    
    return ux, uy, uz

#-------------------------------------------------------------------------------

def cart2pol(x1,x2):
    '''
    Conversion for cartesian (x,y) to polar coordinates (for mogi model).
    '''
    theta = np.arctan2(x2,x1)
    r = np.hypot(x2,x1)
    return theta, r

#-------------------------------------------------------------------------------

def pol2cart(theta,r):
    '''
    Conversion from polar coordinates to cartesian (for mogi model).
    '''
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1,x2
    
#-------------------------------------------------------------------------------

def profile_data(x,y,data,prof_start,prof_end,params):
    
    '''
    Generates a profile through gridded data.
    
    INPUTS:
        data = numpy array of values to profile
        x = vector of coords for the x axis
        y = vector of coords for the y axis
        prof_start = (x, y) pair for the start of the profile line
        prof_end = (x, y) pair for the end of the profile line
        params = dictionary of parameters for the profiler (currently nbins and width)
    
    '''
    
    xx,yy = np.meshgrid(x,y)
    
    prof_start = np.array(prof_start)
    prof_end = np.array(prof_end)
    
    # Profile dimensions relative to profile itself
    prof_dist = np.sqrt((prof_start[1]-prof_end[1])**2 + (prof_start[0]-prof_end[0])**2)
    prof_bin_edges = np.linspace(0, prof_dist ,params["nbins"]+1)    
    prof_bin_mids = (prof_bin_edges[:-1] + prof_bin_edges[1:]) / 2
    
    # Profile points in lat long space
    bin_mids = np.linspace(0,1,params["nbins"]+1)
    bin_grad = prof_end - prof_start
    x_mids = prof_start[0] + (bin_mids * bin_grad[0])
    y_mids = prof_start[1] + (bin_mids * bin_grad[1])
    
    # Gradient of line perpendicular to profile
    bin_grad_norm = (params["width"]/2) * bin_grad / np.linalg.norm(bin_grad)
    
    # Corner points of bins
    bin_x1 = x_mids + bin_grad_norm[1]
    bin_x2 = x_mids - bin_grad_norm[1]
    bin_y1 = y_mids - bin_grad_norm[0]
    bin_y2 = y_mids + bin_grad_norm[0]
    
    # Pre-allocate outputs
    bin_val = np.zeros_like((bin_x1[:-1]))
    bin_std = np.zeros_like(bin_val)
    
    # Trim data set to points inside any bin (improves run time)
    full_poly = path.Path([(bin_x1[0], bin_y1[0]), (bin_x1[-1], bin_y1[-1]), (bin_x2[-1], bin_y2[-1]), (bin_x2[0], bin_y2[0])])
    poly_points = full_poly.contains_points(np.transpose([xx.flatten(),yy.flatten()]))
    poly_points = poly_points.reshape(data.shape)
    trim_data = data[poly_points]
    trim_xx = xx[poly_points]
    trim_yy = yy[poly_points]
    
    # Loop through each bin identifying the points that they contain
    for ii in range(0,params["nbins"]):
                            
        poly_x = np.array([bin_x1[ii], bin_x1[ii+1], bin_x2[ii+1], bin_x2[ii]]);
        poly_y = np.array([bin_y1[ii], bin_y1[ii+1], bin_y2[ii+1], bin_y2[ii]]);
        
        poly = path.Path([(poly_x[0], poly_y[0]), (poly_x[1], poly_y[1]), (poly_x[2], poly_y[2]), (poly_x[3], poly_y[3])])
        
        poly_points = poly.contains_points(np.transpose([trim_xx,trim_yy]))
                            
        in_poly_vals = trim_data[poly_points]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bin_val[ii] = np.nanmean(in_poly_vals)
    
    # get point cloud
    poly_x = np.array([bin_x1[0], bin_x1[-1], bin_x2[-1], bin_x2[0]])
    poly_y = np.array([bin_y1[0], bin_y1[-1], bin_y2[-1], bin_y2[0]])
    points_poly = np.vstack((poly_x,poly_y)).T
    points_poly = np.vstack((points_poly,np.array([points_poly[0,0],points_poly[0,1]])))
    
    poly = path.Path([(poly_x[0], poly_y[0]), (poly_x[1], poly_y[1]), (poly_x[2], poly_y[2]), (poly_x[3], poly_y[3])])
    poly_points = poly.contains_points(np.transpose([trim_xx,trim_yy]))
    points_val = trim_data[poly_points]
    points_x = trim_xx[poly_points]
    points_y = trim_yy[poly_points]
    
    prof_m = (prof_start[1] - prof_end[1]) / (prof_start[0] - prof_end[0])
    points_m = (prof_start[1] - points_y) / (prof_start[0] - points_x)
    points_prof_angle = np.arctan((points_m - prof_m) / (1 + prof_m * points_m))
    points2prof_start = np.sqrt((prof_start[1] - points_y)**2 + (prof_start[0] - points_x)**2)
    points_dist = points2prof_start * np.cos(points_prof_angle)
    
    return bin_val, prof_bin_mids, points_val, points_dist, points_poly


#-------------------------------------------------------------------------------

def plot_comparison(ulos_asc, disp_asc, ulos_desc, disp_desc, xcen, ycen, extents, clim):
    
    '''
    Generates a plot comparing modelled, observed, and residual displacements for two frames.
    Currently the frame names are hard-coded.
    
    INPUTS:
        ulos_asc = ascending model
        disp_asc = ascending observations
        ulos_desc = descending model
        disp_desc = descending observations
        xcen = x coord of mogi centre
        ycen = y coord of mogi centre
        extents = extents of displacement files [xmin, xmax, ymin, ymax]
        clim = colour limits for model and observed
    '''

    # plot model, data, and residual
    fig, axs = plt.subplots(2,3,figsize=(20,17))
    fig.tight_layout(h_pad=10, w_pad = 10)

    # ascending model
    im = axs[0,0].imshow(ulos_asc, extent=extents, vmin=clim[0], vmax=clim[1], cmap=cm.batlow)
    axs[0,0].scatter(xcen, ycen, color='red')
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label='Line-of-sight displacement (mm)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    axs[0,0].set_title('Model displacement', fontsize=20)
    axs[0,0].set_xlabel('x-coord (km)', fontsize=18)
    axs[0,0].set_ylabel('y-coord (km)', fontsize=18)
    axs[0,0].tick_params(labelsize=16)
    axs[0,0].set_xlim(extents[0], extents[1])
    axs[0,0].set_ylim(extents[2], extents[3])
    axs[0,0].annotate('018A_12668_131313', xy=(0, 0.5), xytext=(-axs[0,0].yaxis.labelpad - 40, 0),
                    xycoords=axs[0,0].yaxis.label, textcoords='offset points',
                    fontsize=20, ha='left', va='center', rotation=90, weight='bold')

    # ascending data
    im = axs[0,1].imshow(disp_asc, extent=extents, vmin=clim[0], vmax=clim[1], cmap=cm.batlow)
    axs[0,1].scatter(xcen, ycen, color='red')
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label='Line-of-sight displacement (mm)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    axs[0,1].set_title('Observed displacement', fontsize=20)
    axs[0,1].set_xlabel('x-coord (km)', fontsize=18)
    axs[0,1].tick_params(labelsize=16)
    axs[0,1].set_xlim(extents[0], extents[1])
    axs[0,1].set_ylim(extents[2], extents[3])

    # asceding residual (observed - model)
    im = axs[0,2].imshow(disp_asc-ulos_asc, extent=extents, cmap=cm.batlow)
    axs[0,2].scatter(xcen, ycen, color='red')
    divider = make_axes_locatable(axs[0,2])
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label='Residual displacement (mm)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    axs[0,2].set_title('Residual (observed - model)', fontsize=20)
    axs[0,2].set_xlabel('x-coord (km)', fontsize=18)
    axs[0,2].tick_params(labelsize=16)
    axs[0,2].set_xlim(extents[0], extents[1])
    axs[0,2].set_ylim(extents[2], extents[3])


    # descending model
    im = axs[1,0].imshow(ulos_desc, extent=extents, vmin=clim[0], vmax=clim[1], cmap=cm.batlow)
    axs[1,0].scatter(xcen, ycen, color='red')
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label='Line-of-sight displacement (mm)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    axs[1,0].set_xlabel('x-coord (km)', fontsize=18)
    axs[1,0].set_ylabel('y-coord (km)', fontsize=18)
    axs[1,0].tick_params(labelsize=16)
    axs[1,0].set_xlim(extents[0], extents[1])
    axs[1,0].set_ylim(extents[2], extents[3])
    axs[1,0].annotate('083D_12636_131313', xy=(0, 0.5), xytext=(-axs[1,0].yaxis.labelpad - 40, 0),
                    xycoords=axs[1,0].yaxis.label, textcoords='offset points',
                    fontsize=20, ha='left', va='center', rotation=90, weight='bold')

    # descending data
    im = axs[1,1].imshow(disp_desc, extent=extents, vmin=clim[0], vmax=clim[1], cmap=cm.batlow)
    axs[1,1].scatter(xcen, ycen, color='red')
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label='Line-of-sight displacement (mm)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    axs[1,1].set_title('Observed displacement', fontsize=20)
    axs[1,1].set_xlabel('x-coord (km)', fontsize=18)
    axs[1,1].tick_params(labelsize=16)
    axs[1,1].set_xlim(extents[0], extents[1])
    axs[1,1].set_ylim(extents[2], extents[3])

    # descending residual (observed - model)
    im = axs[1,2].imshow(disp_desc-ulos_desc, extent=extents, cmap=cm.batlow)
    axs[1,2].scatter(xcen, ycen, color='red')
    divider = make_axes_locatable(axs[1,2])
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label='Residual displacement (mm)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    axs[1,2].set_title('Residual (observed - model)', fontsize=20)
    axs[1,2].set_xlabel('x-coord (km)', fontsize=18)
    axs[1,2].tick_params(labelsize=16)
    axs[1,2].set_xlim(extents[0], extents[1])
    axs[1,2].set_ylim(extents[2], extents[3])

    plt.show()