#!/usr/bin/env python
# reads scatterers picked by hand (json file) in vespagrams and finds scatterer locations using SWAT.

import csv
import taup
from scattererwhereartthou import SWAT, mapplot, sliceplot
import sys,re,os
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
sys.path.append("../")
import json
from scipy.spatial import Delaunay
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cmcrameri import cm
##
def loadScatterers(sct_json):
    with open(sct_json, "r") as file:
        scatterers = json.load(file)
        return scatterers

def fibonacci_sphere(number_points):
    #https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    phi = np.pi * (np.sqrt(5.) - 1.)

    i = np.arange(number_points)
    y = 1 - 2 * i / (number_points - 1)
    radius = np.sqrt(1 - y**2)
    theta = phi * i

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.column_stack((x, y, z))
#
def create_Fib_grid(delta_deg=1,depth_delta=100):
    """
    for a delta_deg2 area, creates a fibonacci_sphere for each depth (depth_delta).
    returns fib_grid.
    the number of points at each depth slice change such that the area is conserved.
    """
    R = 6371
    #area per point at surface
    area_point = (np.radians(delta_deg) * R)**2

    radii = np.arange(2900, 6370, depth_delta)
    n_points = np.round(4 * np.pi * radii**2 / area_point).astype(int)

    fib_grid = []
    for r, n in zip(radii, n_points):
        fib = fibonacci_sphere(n)
        fib_grid.append(fib * r)

    fib_grid = np.vstack(fib_grid)

    return fib_grid
##
def swat_sct_volume(taupserver,scatterers,i,scat,figname=None):
    """
    read one sct at a time. using swat, finds potential sctrs.
    finds delta time/baz/slow from picked vals.
    for the scatteres, finds a convex volume.
    the volume caluclate is done in cartesian coordinates.
    around lat0 and lon0 (mean lat long of all scat locations.)
    """
    model="iasp91"
    phase="P"   # reference phase
    max_dist_step=2.0 # max separation between path scatterers in degrees, default is 2 deg
    min_dist_step=0.05
    evt=(scatterers['SRC_LAT'] ,scatterers['SRC_LON'])
    eventdepth=(scatterers['SRC_DEP'])
    sta=(scatterers['REC_LAT'] ,scatterers['REC_LON'])
    ######
    slow_sct=scat['SCAT_slow_max']
    time_sct=round(scat['SCAT_time_max'],2)
    if float(scatterers['baz_offset']) > 0.5 :
        print(f'Baz offset in Eq-array pair:{float(scatterers['baz_offset'])}')
        print('subtracting it from the observed Backazimuth of scatterer..')
        baz_sct=scat['SCAT_baz_max']-float(scatterers['baz_offset'])
    else:
        baz_sct=scat['SCAT_baz_max']

    sc_time_delta=round(max(scat['SCAT_sl_time_5_delta'],scat['SCAT_bz_time_5_delta'],3),2)
    sc_slow_delta=round(max(scat['SCAT_slow_5_delta'], .1),2)
    sc_baz_delta= max(scat['SCAT_baz_5_delta'], 1)
    #
    print("Scatterer props..")
    print(f"Delta time/slow/baz used: {sc_time_delta}sec, {sc_slow_delta}sec/deg, {sc_baz_delta}deg")
    ###
    bazdelta=sc_baz_delta/2

    sta_scat_revphase="P,Ped,PP,PS" ###
    # evt_scat_phase="p,s,P,S,Ped,Sed,pP,sP,pS,sS,PP,SS,SP,PS"

    sta_scat_revphase='P,Ped,PP'
    evt_scat_phase='p,P,Ped'

    swatList = []
    swat = SWAT(taupserver, eventdepth, model=model,
        sta_scat_revphase=sta_scat_revphase,
        evt_scat_phase=evt_scat_phase)
    swat.event(*evt)
    swat.station(*sta)
    swat.max_dist_step = max_dist_step
    swat.min_dist_step = min_dist_step

    baz_GCP=swat.es_baz

    slow_list=[slow_sct-sc_slow_delta/2,slow_sct,slow_sct+sc_slow_delta/2]
    time_list=[time_sct-sc_time_delta/2,time_sct,time_sct+sc_time_delta/2]
    print(f"slow: {slow_list}, traveltimes: {time_list}, bazOff:{baz_sct}, bazdelta:{bazdelta}")
    # for i,sl in enumerate(slow_list):
    ans = swat.find_via_path(slow_list, time_list, bazoffset=baz_sct, bazdelta=bazdelta)
    print(f"Length of sct: {len(ans.scatterers)}")#", for sl:{sl}, time:{time}")
    swatList.append(ans)


    len_all=0
    sct_loc=[]
    for SctDist in swatList:
        len_all+=len(SctDist.scatterers)
        for sct in SctDist.scatterers:
            # print(f"")
            # print(f"slow:{sct.sta_scat_rayparam}, total_time:{sct.scat.time+sct.evt_scat.time:.2f}, baz: {sct.scat_baz-baz_GCP:.2f}")
            # print(f"Phase: {sct.evt_scat.phase} & {sct.sta_scat_phase}. Lat, Long, depth:{sct.scat.lat:.4f}, {sct.scat.lon:.4f}, {sct.scat.depth:.4f}")
            sct_loc.append((sct.scat.lat,sct.scat.lon,sct.scat.depth))
        #
    return swatList,sct_loc
#
def get_hull_volume(points):
    """
    input: swat output (scatterer locations lat lon depth)
    calculates a convex hull that fits all points.
    """
    # Plot sctrs in 3D as x y z.
    points = np.asarray(points)
    lat = points[:, 0]
    lon = points[:, 1]
    depth = points[:, 2]
    # converting to earth centered cartesian..
    R = 6371
    r = R - depth

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    X = r * np.cos(lat_rad) * np.cos(lon_rad)
    Y = r * np.cos(lat_rad) * np.sin(lon_rad)
    Z = r * np.sin(lat_rad)

    xyz = np.column_stack((X, Y, Z))
    hull = ConvexHull(xyz)
    print(f"Convex hull volume: {hull.volume:.2f} km³ / {hull.volume/(111.32**3):.2f} degree³")

    return xyz,hull,depth
#
def plot_hull_3d(xyz,depth,hull,figname=None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])

        ax.plot(xyz[simplex, 0],xyz[simplex, 1],xyz[simplex, 2],"k-",linewidth=0.8,alpha=0.25)
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],c=depth,cmap="viridis",s=50,edgecolor="k")

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    ax.invert_zaxis()
    ax.view_init(elev=-20, azim=-35,roll=10)
    cbar = fig.colorbar(sc, ax=ax, pad=0.1,shrink=0.5,fraction=.15)
    cbar.set_label("Depth (km)")
    ax.set_title(f"Volume: {hull.volume:.2f} km³ / {hull.volume/(111.32**3):.2f} deg³. # sct:{len(depth)}")
    plt.tight_layout()
    if figname:
        plt.savefig(figname,dpi=300,bbox_inches='tight', pad_inches=0.1)
    plt.show()
#
def find_weights_Scatterer(hull_convex,fib_grid,tolerance):
    # weights, inside = hull_to_bin_weights(hull_convex,lat0,lon0)

    weights = np.zeros(len(fib_grid))

    """
    ConvexHull.equations, stores equations for every triangular face which forms the volume.
    they are of the form aX+bY+cZ+d=0, with shape (number_of_faces, 4).
    we take matrix multiplication fo fib_grid and first three rows and add last column.
    e-8 is to inlcude point which are really close to the hull.
    summary: For each Fibonacci point, evaluate the plane equation for every face of the convex hull.
    If the point satisfies the inside condition for every face, mark it True, otherwise mark it False.
    tolerance is in km.
    For convex hull, all planes are deined such that inside the hull is negative for the points.
    """
    inside = np.all(fib_grid @ hull_convex.equations[:, :-1].T+ hull_convex.equations[:, -1] <= tolerance,axis=1)

    n_inside=inside.sum()
    print(f"Number of cells inside hull: {n_inside}")

    if n_inside>0:
        weights[inside] = 1/n_inside

    return weights
    # counts, edges = np.histogramdd(samples,bins=[lat_edges, lon_edges, dep_edges])

def plot_weights_map(fib_grid,total_weights,color_by='Weight',dmin=None,dmax=None,figname=None):
    """
    map view plot for scatterers using fib_grid.
    can color by weights or depth.
    can mask based on depth such that id dmin/dmax (0,500), we basically stack all at one depth.
    if not, then all depths are plotted at surface.
    """
    R = 6371
    X, Y, Z = fib_grid.T
    r = np.sqrt(X**2 + Y**2 + Z**2)
    lat = np.degrees(np.arcsin(Z / r))
    lon = np.degrees(np.arctan2(Y, X))

    depth = R - r
    ##
    if dmin:
        dmin = dmin
        dmax = dmax
        mask = (depth >= dmin) & (depth < dmax)
        lat=lat[mask]
        lon=lon[mask]

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
    pc_pacific = ccrs.PlateCarree(central_longitude=180)
    ax.set_extent((-60, 60, -35, 75), crs=pc_pacific)

    # ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightblue',alpha=.2, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="1.0", zorder=0)

    if color_by == 'Weight':
        sc = ax.scatter(lon,lat,c=total_weights,s=25,cmap="cividis",\
        transform=ccrs.PlateCarree(),alpha=0.8)
        cbar = plt.colorbar(sc, ax=ax, pad=0.03)
        cbar.set_label(color_by)
    else:
        sc = ax.scatter(lon,lat,c=depth,s=25,cmap="cmc.turku",\
        transform=ccrs.PlateCarree(),alpha=0.99)
        cbar = plt.colorbar(sc, ax=ax, pad=0.03)
        cbar.set_label(color_by)

    if figname:
        plt.savefig(figname,dpi=300,bbox_inches='tight', pad_inches=0.1)

    plt.show()
#
def plot_sctsFib_3d(fib_grid,total_weights,lats_path,lons_path,depths_path,delta_deg=0.5,depth_delta=50,figname=None):
    lat_max=25
    depth_max=2000
    R = 6371
    X, Y, Z = fib_grid.T
    r = np.sqrt(X**2 + Y**2 + Z**2)
    lat = np.degrees(np.arcsin(Z / r))
    lon = np.degrees(np.arctan2(Y, X))

    depth = R - r
    # Volume represented by each Fibonacci point
    area_point = (np.radians(delta_deg) * R)**2
    point_volume = area_point * (r / R)**2 * depth_delta
    size_scale = 400
    scaled_size = size_scale * point_volume / point_volume.max()
    ##
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(lon,lat,depth,c=total_weights,s=scaled_size,marker='o',cmap="cividis",alpha=1)#,edgecolor="white")
    # plot path
    lon_360 = np.asarray(lons_path) % 360
    lats_path=np.asarray(lats_path)
    depths_path=np.asarray(depths_path)

    mask = (lats_path <= lat_max) & (depths_path <= depth_max)
    ax.plot(lon_360[mask],lats_path[mask],depths_path[mask],ls='-',lw=1.6,c='brown')
    # cbar = plt.colorbar(sc, ax=ax, pad=0.03)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth (km)")
    ax.set_ylim(ymax=lat_max)
    ax.set_zlim(zmax=depth_max)
    ax.invert_zaxis()
    # ax.view_init(elev=-20, azim=-35,roll=10)
    cbar = fig.colorbar(sc, ax=ax, pad=0.1,shrink=0.5,fraction=.15)
    cbar.set_label("Weight")
    plt.tight_layout()
    if figname:
        plt.savefig(figname,dpi=300,bbox_inches='tight', pad_inches=0.1)
    plt.show()

def get_rp_using_taup(taupserver,model,phase,scatterers):
    evt=[scatterers['SRC_LAT'],scatterers['SRC_LON']]
    sta=[scatterers['REC_LAT'],scatterers['REC_LON']]
    params = taup.PathQuery()
    params.phase(phase)
    params.model(model)
    params.event(*evt)
    params.station(*sta)
    params.sourcedepth([scatterers['SRC_DEP']])
    # params.degree(delta_deg_val)
    pathResult = params.calc(taupserver)
    lats_path=[]
    lons_path=[]
    depths_path=[]
    for a in pathResult.arrivals:
        for pathseg in a.path:
            for td in pathseg.segment:
                lats_path.append(td.lat)
                lons_path.append(td.lon)
                depths_path.append(td.depth)

    if not pathResult.arrivals:
        return None
    return lats_path,lons_path,depths_path
#
# def main():
grd_num=114
taup_path="~/Research/sct_wat/TauP/build/install/TauP/bin/taup"
# taup_path=None

sct_json="sac_files/220914_110406/py_picks/grid_num_{}_2022914114_PICKS.json".format(grd_num)
# sys.exit()

### bin edges..
scatterers = loadScatterers(sct_json)
with taup.TauPServer(taup_path=taup_path) as taupserver:
    lats_path,lons_path,depths_path = get_rp_using_taup(taupserver,'iasp91', "P", scatterers)
    fib_grid=create_Fib_grid(delta_deg=.5,depth_delta=50)
    total_weights = np.zeros(len(fib_grid))
    for i, scat in enumerate(scatterers['sct']):
        figname='220914_{}_{}_P_hull.png'.format(grd_num,i)
        swatList,sct_loc=swat_sct_volume(taupserver,scatterers,i, scat)
        xyz,hull_convex,depth=get_hull_volume(sct_loc)
        plot_hull_3d(xyz,depth,hull_convex,figname=figname)
        weights=find_weights_Scatterer(hull_convex,fib_grid,tolerance=10)
        total_weights += weights
        print(f'Scatterer#{i+1} done')
        print("------------------\n")
    #########
    print(f'sum of all weights:{total_weights.sum()}')
##
# sys.exit()

## just keep non zero fib grid and weights..

nonzero = total_weights > 0
fib_grid_nz = fib_grid[nonzero]
t_weights_nz = total_weights[nonzero]
#
# color_by options: Weight or Depth (km)
color_by = input("Choose either to color scatterers by 'Weight' or 'Depth'. Type exact and hit return: ")
# color_by='Weight'
# color_by='Depth'
plot_weights_map(fib_grid_nz,t_weights_nz,color_by=color_by,figname='map_{}_{}.png'.format(grd_num,color_by))
plot_sctsFib_3d(fib_grid_nz,t_weights_nz,lats_path,lons_path,depths_path,delta_deg=0.5,depth_delta=50,figname='volume_{}.png'.format(grd_num))

##
