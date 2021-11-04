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
from osgeo import gdal

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


def cart2pol(x1,x2):
    '''
    Conversion for cartesian (x,y) to polar coordinates.
    '''
    theta = np.arctan2(x2,x1)
    r = np.hypot(x2,x1)
    return theta, r


def pol2cart(theta,r):
    '''
    Conversion from polar coordinates to cartesian.
    '''
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1,x2
    
    
#-------------------------------------------------------------------------------

def load_geotif(file_name):
    '''
    Loads a geotif file using gdal and calculate the x and y coordinates.
    
    INPUTS:
        fie_name = path to file (str)
    
    OUTPUTS:
        data = numpy array of geotif content
        x = vector of x-coords
        y = vector of y-coords
    '''
    
    # open tif
    geotiff = gdal.Open(file_name)
    
    # read to numpy array
    data = geotiff.ReadAsArray()
    
    # calculate x and y corners
    ulx, xres, xskew, uly, yskew, yres  = geotiff.GetGeoTransform()
    lrx = ulx + (geotiff.RasterXSize * xres)
    lry = uly + (geotiff.RasterYSize * yres)
    
    # create coord vectors
    x = np.arange(ulx, lrx+xres, xres)
    y = np.arange(lry, uly+yres, yres)
    
    return data, x, y
    
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