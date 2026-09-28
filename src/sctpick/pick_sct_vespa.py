#when picking scatteres, the left click should be high slow/baz and right click low slow/baz!!!
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
import pygmt
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as mcolors
# from cmcrameri import cm
# import ScientificColourMaps8 as SCM8
import matplotlib.cm as cmm
from cmaptools import readcpt, joincmap, DynamicColormap
import glob as glob
import re
from obspy.core import UTCDateTime as utc
# from IPython import get_ipython
from obspy.taup.taup_geo import calc_dist,calc_dist_azi
from obspy.taup import TauPyModel
import subprocess
from matplotlib.widgets import CheckButtons
import xarray as xr
import scipy.signal as sci
import warnings
import scipy.stats as stats
from scipy.stats import norm, skewnorm, kurtosis
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import json
####
def extract_gridnumber(filename):
    match = re.search(r'gridnum(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None
####
def extract_datapackfile(grid_number,folder_datapack):
    for file_name in os.listdir(folder_datapack):
        if file_name.endswith('.txt'):
            grid_num=extract_gridnumber(file_name)
            if grid_num==grid_number:
                # print(file_name)
                return file_name
###
def extract_region_from_grid(grid_number,grid_folder,type,beam_type):
    file_pattern = os.path.join(grid_folder, '{}_{}_grid_{}*.grd'.format(beam_type,type,grid_number))

    for filename in glob.glob(file_pattern):
        # print(filename)
        grd_file = pygmt.load_dataarray(filename)
        grd_info=pygmt.grdinfo(filename,per_column=True)
        grd_info=grd_info.split()
        # print(grd_info)
        region_type=[float(grd_info[0])+10, float(grd_info[1])-10, float(grd_info[2]), float(grd_info[3]),float(grd_info[4]),float(grd_info[5])]
        return(grd_file,region_type)
######
def calc_tt(eq_lat,eq_long,st_lat,st_long,eq_depth):
    model = TauPyModel(model="ak135")
    dist=calc_dist(eq_lat,eq_long,st_lat,st_long,6400,0)
    arr_PP=arr_pP=arr_sP=arr_pPP=float('nan')

    try:
        arr_P = model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["P"])[0]
    except:
        arr_P = model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["Pdiff"])[0]
    try:
        arr_PP = model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["PP"])[0]
    except:
        print('PP phase didnt arrive')
    try:
        arr_pP=model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["pP"])[0]
    except:
        arr_pP=model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["pPdiff"])[0]
    try:
        arr_sP=model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["sP"])[0]
    except:
        arr_sP=model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["sPdiff"])[0]

    arr_pPP=model.get_travel_times(source_depth_in_km=eq_depth,distance_in_degree=dist,phase_list=["pPP"])[0]

    return(arr_P,arr_PP,arr_pP,arr_sP,arr_pPP)

def extract_grid_list(grid_folder):
    file_pattern = os.path.join(grid_folder, 'xf_slow_grid_*.grd')
    files = glob.glob(file_pattern)

    # List to store the extracted grid numbers
    grid_numbers = []

    # Regular expression to extract numbers after 'grid_'
    pattern = re.compile(r'grid_(\d+)_')

    for filename in files:
        match = pattern.search(filename)
        if match:
            grid_number = int(match.group(1))  # Convert to integer if needed
            grid_numbers.append(grid_number)
    grid_numbers.sort()
    return grid_numbers

##
def set_locators(ax, axis_type='default'):
    if axis_type == 'slow':
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(1))
    elif axis_type == 'baz':
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(10))
    else:
        raise ValueError("Unsupported axis type: {}. Use 'default' or 'alt'.".format(axis_type))

def extract_grid_nums(main_folder):
    pattern = os.path.join(main_folder, '*.jpg')

    # List to store the gridnum values
    gridnum_list = []
    for file_path in glob.glob(pattern):
    # Extract the filename from the file path
        filename = os.path.basename(file_path)

        # Use regex to find the number after 'gridnum'
        match = re.search(r'gridnum(\d+)', filename)
        if match:
            gridnum_list.append(int(match.group(1)))

    return gridnum_list

def get_max_Z(xarray, time_step=5,overlap=2.5):

    z=xarray
    window_size = time_step
    step_size = overlap
    num_x_slices = len(z.x)

    midpoints = []
    max_values = []
    y_values = []

    for start_idx in np.arange(0, num_x_slices - window_size + 1, step_size):
        slice_x = z.isel(x=slice(int(start_idx), int(start_idx + window_size)))
        max_val = slice_x.max(dim=['y', 'x'])
        max_idx_y = slice_x.where(slice_x == max_val, drop=True).y
        if len(max_idx_y) >1: #sometimes, the index returned more than one position..
            max_idx_y=max_idx_y[0]

        # Store the midpoint x, max z value, and corresponding y value
        midpoints.append(slice_x.coords['x'].mean().item())
        max_values.append(max_val.values.max())
        y_values.append(max_idx_y.item())

    # Convert lists to numpy arrays for easier manipulation if needed
    midpoints = np.array(midpoints)
    max_values = np.array(max_values)
    y_values = np.array(y_values)

    return(midpoints,y_values,max_values)


def extract_max_coher_clicks(grd,clicks):
    #for a grd array, extracts the max coherence for the manually
    # clicked crosses on vespa
    xf_pick=[]
    xf_pick_95=[]
    for i in range(0, len(clicks[0]), 2):
        # Extract two rows at a time
        rows_s = clicks[0][i:i+2]
        time_st,time_end=rows_s[0][0],rows_s[1][0]

        sl_max,sl_min=rows_s[0][1],rows_s[1][1]
        masked_array_slow,slow_values,time_vals=get_95percent_in_box(grd=grd,percent=.95,x_min=time_st,x_max=time_end,y_min=sl_min,y_max=sl_max)

        print(f"click {i}, min/max= {slow_values.min():.3f}, {slow_values.max():.3f} ")
        xf_pick_95.append([slow_values.max()-slow_values.min(),time_vals.max()-time_vals.min()])

        slow_xf_pick = grd.where((grd.x > time_st) & (grd.x < time_end), drop=True)
        max_index_s = slow_xf_pick.argmax().item()
        max_coords_slow = np.unravel_index(max_index_s, slow_xf_pick.shape) ##gets x,y position of Z max in slow_xf_pick
        xf_pick_slow = [slow_xf_pick.x[max_coords_slow[1]].item(), slow_xf_pick.y[max_coords_slow[0]].item(), slow_xf_pick.max().item()]

        xf_pick.append(xf_pick_slow)
    return xf_pick,xf_pick_95

def get_95percent_in_box(grd,percent=.95,x_min=None,x_max=None,y_min=None,y_max=None):
    """
    for a grd, using the chosen box on vespa for each scatter, find the points around max with 95% of max Z val.
    """
    window_data_t = grd.sel(x=slice(x_min, x_max))
    window_data = window_data_t.sel(y=slice(y_min, y_max))
    zmax = window_data.max().item()

    threshold_value = percent * zmax

    # masking using Z > 95% of max
    masked_array = window_data.where(window_data >= threshold_value)

    y_values = masked_array['y'].values[masked_array.notnull().any(dim='x')]
    x_values = masked_array['x'].values[masked_array.notnull().any(dim='y')]

    return masked_array, y_values, x_values

def get_peaks_grd(grd):

    midpoints,y_values,max_values=get_max_Z(grd,time_step=5,overlap=2.5)
    # midpoints_slow,y_values_slow,max_values_slow=get_max_Z(slow_grd_curtail,time_step=5,overlap=2.5)
    midpoints = np.array(midpoints)
    max_values = np.array(max_values)
    y_values = np.array(y_values)

    # get peaks in max_values (baz and slow grds)
    indexes, dict = sci.find_peaks(np.array(max_values),height=.2*np.max(max_values),prominence=.1*np.max(max_values))
    # indexes_slow, dict_slow = sci.find_peaks(np.array(max_values_slow),height=.25*np.max(max_values_slow),prominence=.1*np.max(max_values_slow))
    return midpoints,max_values,y_values,indexes

###
def plot_vespa_pick_slow(folder_pattern,grid_num=None,clicker_onoff=True,plot_amp_factor=1):

    cptfile_='cptfiles/blue-yellow.cpt'#
    cptfile='cptfiles/green-purple-d09.cpt'#

    cmap_try= readcpt(cptfile)
    cmap_slow= readcpt(cptfile_)

    matching_folders = glob.glob(folder_pattern)

    plt.ion()
    plot_amp_factor=plot_amp_factor
    plot_amp_factor_curtail = 1
    plt.rcParams.update({'font.size': 14})
    clicker_onoff=True
    for folder in matching_folders:
        main_folder=folder+'/'

        folder_datapack=main_folder+'data_pack/'
        grid_folder=main_folder+'grid_folder'
        pick_folder=main_folder+'py_picks/'
        py_figs=main_folder+'py_figs_new/'
        os.makedirs(py_figs, exist_ok=True)

        print('Main folder:',main_folder)
        gridnum_list=extract_grid_nums(main_folder)
        gridnum_list.sort()

        grid_baz_offset=[]
        grid_baz_offset_low_slow=[]
        grid_baz_offset_high_slow=[]

        if grid_num:
            grid_run=[grid_num]
        else:
            grid_run=gridnum_list
        # for grid_number in gridnum_list:
        for grid_number in grid_run:

            print('plot_amp_factor=',plot_amp_factor)
            print('grid_number=',grid_number)

            grid_list=extract_grid_list(grid_folder)

            #################
            beam_deets=folder_datapack+extract_datapackfile(grid_number,folder_datapack)
            print(beam_deets,'\n')
            patterns = {
                "Origin": re.compile(r"Origin: (\d+) (\d+) (\d+) (\d+):(\d+)"),
                "ArrCen": re.compile(r"ArrCen la/lo/elv: (\d+\.\d+) (-?\d+\.\d+) (\d+) Nst:(\d+)"),
                "ArrBaseStn": re.compile(r"ArrBaseStn: (\w+), grid la/lp (\d+), (-?\d+)"),
                "Event": re.compile(r"Event la/lo/dp: (-?\d+\.\d+) (-?\d+\.\d+) (\d+\.\d+)"),
                "Dist": re.compile(r"Dist: (\d+\.\d+)"),
                "Baz": re.compile(r"Baz \(Arr-Evt\): (\d+\.\d+)"),
                "Frequencies": re.compile(r"Frequencies: (\.\d+) - (\.\d+) Hz"),
                "TrcesSNR": re.compile(r"TrcesSNR mn,SD,min,max: (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)"),
                "PredPP": re.compile(r"Pred PP \(prem\) time/U: (\d+\.\d+) (\d+\.\d+)")
            }

            # dictionary to store the extracted values
            deets = {
                "Origin": [],
                "ArrCen": [],
                "ArrBaseStn": [],
                "Event": [],
                "Dist": [],
                "Baz": [],
                "Frequencies": [],
                "TrcesSNR": [],
                "PredPP": []
                }

            # Read the file and match lines with the defined patterns
            with open(beam_deets, 'r') as file:
                for line in file:
                    for key, pattern in patterns.items():
                        match = pattern.search(line)
                        if match:
                            # Handle ArrBaseStn separately to exclude non-numeric values
                            if key == "ArrBaseStn":
                                # Convert only numeric values, excluding the first group (station name)
                                deets[key].extend(match.groups()[1:])
                            else:
                                deets[key].extend(map(float, match.groups()))

            # Convert numeric strings to floats for ArrBaseStn
            deets["ArrBaseStn"] = [float(x) for x in deets["ArrBaseStn"]]
            print('-------------\n')
            print(deets["ArrCen"])

            #------------------------
            # interp_slow=[0.1,0.05]
            # interp_baz="0.1/0.5"
            slow_grd,region_slow=extract_region_from_grid(grid_number,grid_folder,'slow','xf')
            # print(region_slow)
            #####
            baz_grd,region_baz=extract_region_from_grid(grid_number,grid_folder,'baz','xf')
            # print(region_baz)
            ####
            slow_grd_bm,region_slow_bm=extract_region_from_grid(grid_number,grid_folder,'slow','beam')
            # print(region_slow_bm)
            #####
            baz_grd_bm,region_baz_bm=extract_region_from_grid(grid_number,grid_folder,'baz','beam')
            ##
            for line in open(beam_deets,'r'):
                line=line.split()
                if line[6]=='Pred':
                    PP_t_s=[float(line[10]),float(line[11])]

            ###
            try:
                arr_P,arr_PP,arr_pP,arr_sP,arr_pPP=calc_tt(deets['Event'][0],deets['Event'][1],deets['ArrCen'][0],deets['ArrCen'][1],deets['Event'][2])
            except:
                print('one or more phases didnt arrive')

            # Colormaps
            # -------------------------
            cmap_lip = cmm.PuBu
            colA = cmap_lip(np.arange(cmap_lip.N))
            sm_alpha = ListedColormap(colA)

            max_position = baz_grd.argmax(dim=['y', 'x'])
            max_position_slow = slow_grd.argmax(dim=['y', 'x'])

            # Extract the indices for 'y' and 'x'
            y_max = baz_grd['y'][max_position['y']].item()
            x_max = baz_grd['x'][max_position['x']].item()

            x_max_slow = slow_grd['x'][max_position_slow['x']].item()
            y_max_slow = slow_grd['y'][max_position_slow['y']].item()

            print("----------------------\n")
            print(f"Max baz_grd for grid {grid_number} is at time: {x_max:.2f}s, baz: {y_max}")
            print("----------------------\n")

            if  arr_PP.time - arr_sP.time < 30:
                print('very little time between sP and PP')
                break

            slow_grd_curtail = slow_grd.where(
                (slow_grd.x > arr_sP.time + 30) & (slow_grd.x < arr_PP.time - 10),drop=True)
            baz_grd_curtail = baz_grd.where(
                (baz_grd.x > arr_sP.time + 30) & (baz_grd.x < arr_PP.time - 10),drop=True)

            V_max_curtail = baz_grd_curtail.max(dim=['x', 'y'])

            # Peaks
            # -------------------------
            midpoints, max_values, y_values, indexes = get_peaks_grd(baz_grd_curtail)
            midpoints_slow, max_values_slow, y_values_slow, indexes_slow = get_peaks_grd(slow_grd_curtail)
            #
            # Coherence statistics
            # -------------------------
            avg_coherence = baz_grd.mean(dim=['x', 'y'])
            std_coherence = baz_grd.std(dim=['x', 'y'])
            z_values_coh = baz_grd.values.flatten()
            max_mean = baz_grd.max(dim=['x', 'y']) / avg_coherence

            if max_mean.item() < 15:
                print("Max/mean less than 15; skipping this \n")
                continue
            # -------------------------
            # Discrete norms
            # -------------------------
            num_bins = 8
            norm = mcolors.BoundaryNorm(np.linspace(-4, 4, num_bins + 1), cmap_try.N)
            norm_slow = mcolors.BoundaryNorm(np.linspace(1, 9, num_bins + 1), cmap_slow.N)

            fig = plt.figure(figsize=(15, 8))
            # [left, bottom, width, height]
            ax1 = fig.add_axes([0.07, 0.6, 0.38, 0.3])   # slow main
            ax2 = fig.add_axes([0.52, 0.6, 0.38, 0.3])   # baz main
            ax3 = fig.add_axes([0.75, .91, .15, 0.012])    # colorbar (coherence)
            ax4 = fig.add_axes([0.07, 0.27, 0.38, 0.27]) # slow curtailed
            ax5 = fig.add_axes([0.52, 0.27, 0.38, 0.27]) # baz curtailed
            ax7 = fig.add_axes([0.52, 0.1, 0.475, 0.15]) # baz peaks
            ax8 = fig.add_axes([0.07, 0.1, 0.475, 0.15]) # slow peaks
            ax9 = fig.add_axes([0.9, 0.6, 0.07, 0.3])    # histogram

            ## slowness main plot
            slow_grd.plot(
            ax=ax1, cmap=sm_alpha, add_colorbar=False,
            vmin=0, vmax=region_baz[5] / plot_amp_factor, mouseover=True)
            slow_grd.plot.contour(
                ax=ax1, cmap='Greys_r', linewidths=.65, add_colorbar=False,
                levels=np.linspace(region_baz[5] / 8, region_baz[5] / plot_amp_factor, 4))

            ax1.axvline(x=(arr_sP.time+30), color='darkorange', linestyle='--', lw=1.3)
            ax1.axvline(x= (arr_PP.time - 10), color='darkorange', linestyle='--', lw=1.3)
            ax2.axvline(x=(arr_sP.time+30), color='darkorange', linestyle='--', lw=1.3)
            ax2.axvline(x= (arr_PP.time - 10), color='darkorange', linestyle='--', lw=1.3)

            for phase in [arr_pP,arr_sP,arr_PP,arr_pPP]:

                if 'diff' in phase.name:
                    ax1.scatter(phase.time,phase.ray_param*0.0174533,marker='o',c='CORNFLOWERBLUE',s=50,edgecolors='white',zorder=10)
                    ax1.text(phase.time, 1.5+phase.ray_param * 0.0174533, phase.name, bbox={'facecolor': 'white', 'alpha': 0.85, 'pad': 1.5},fontsize=14,c='CORNFLOWERBLUE', rotation='vertical',ha='center')
                else:
                    ax1.scatter(phase.time,phase.ray_param*0.0174533,marker='o',c='violet',s=50,edgecolors='white',zorder=10)
                    ax1.text(phase.time, 1.5+phase.ray_param * 0.0174533, phase.name, bbox={'facecolor': 'white', 'alpha': 0.85, 'pad': 1.5},fontsize=14,c='violet', rotation='vertical',ha='center')

            #### ax2: baz main

            baz_grd.plot(ax=ax2, cmap=sm_alpha, add_colorbar=False,
                vmin=0, vmax=region_baz[5] / plot_amp_factor, mouseover=True)
            baz_grd.plot.contour(ax=ax2, cmap='Greys_r', linewidths=.65, add_colorbar=False,
                levels=np.linspace(region_baz[5] / 8, region_baz[5] / plot_amp_factor, 4))

            ax2.axhline(y=0, color='darkred', linestyle='--')
            ax2.scatter(x_max, y_max, marker='d', c='darkred', s=55,
                        edgecolors='white', zorder=10)

            ax2.text(region_baz[0]-10, 26,
                f'max ({int(baz_grd.max().item())}) at {y_max}$^\\circ$ Backazimuth',
                c='darkred', size=12,bbox={'facecolor': 'white', 'alpha': 0.85, 'pad': 1.5})

            ### curtailed slow baz ## ax4/5

            slow_grd_curtail.plot(ax=ax4, cmap=sm_alpha, add_colorbar=False,
                vmin=0, vmax=V_max_curtail / plot_amp_factor_curtail)
            slow_grd_curtail.plot.contour(ax=ax4, cmap='Greys_r', linewidths=.65, add_colorbar=False,
                levels=np.linspace(V_max_curtail / 8, V_max_curtail / plot_amp_factor_curtail, 4))
            ###
            baz_grd_curtail.plot(ax=ax5, cmap=sm_alpha, add_colorbar=False,
                vmin=0, vmax=V_max_curtail / plot_amp_factor_curtail)
            baz_grd_curtail.plot.contour(ax=ax5, cmap='Greys_r', linewidths=.65, add_colorbar=False,
                levels=np.linspace(V_max_curtail / 8, V_max_curtail / plot_amp_factor_curtail, 4))

            ###
            ax5.axhline(y=y_max, color='darkred', linestyle='-',lw=1.2)
            ax4.axhline(y=arr_pP.ray_param_sec_degree, color='black', linestyle='--',lw=1)
            ax4.axhline(y=arr_PP.ray_param_sec_degree, color='black', linestyle='--',lw=1)

            ## ax7/8 peaksss

            ax7.plot(midpoints, max_values, '-', lw=.25, c='black', alpha=.65)
            ax8.plot(midpoints_slow, max_values_slow, '-', lw=.25, c='black', alpha=.65)

            scatter_7 = ax7.scatter(midpoints, max_values, c=y_values,
                cmap=cmap_try, norm=norm,edgecolor='white', s=15, alpha=.88, linewidth=.15)
            scatter_8 = ax8.scatter(midpoints_slow, max_values_slow, c=y_values_slow,
                cmap=cmap_slow, norm=norm_slow,edgecolor='white', s=15, alpha=.88, linewidth=.15)

            # saving based on mean max
            ax7.scatter(midpoints[indexes], max_values[indexes],
                        marker='+', c='black', s=40, lw=1.25)
            ax8.scatter(midpoints[indexes], max_values[indexes],
                            marker='+', c='black', s=40, lw=1.25)

            ##### ax9 histo

            ax9.hist(z_values_coh, density=True, bins='auto',
             histtype='stepfilled', alpha=0.65, color='orchid')
            ax9.axvline(avg_coherence, color='slateblue', linestyle='--', lw=1.25)

            ax9.set_xlim(0, baz_grd_curtail.max().item() / 5)
            ax9.set_yticks([])
            ax9.set_xticks([])

            fig.text(0.94, .75, f'mean={avg_coherence.item():.1f}',fontsize=11, ha='center', color='slateblue')
            fig.text(0.94, .725, f'std={std_coherence.item():.1f}',fontsize=11, ha='center')
            fig.text(0.94, .70, f'm/m={max_mean.item():.1f}',fontsize=11, ha='center')

            ### Colorbars, grids, labels, limits

            sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=region_baz[5] / plot_amp_factor),
                cmap=sm_alpha)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=ax3, orientation='horizontal', extend='max')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.xaxis.tick_top()
            cbar.ax.set_xlabel('Coherence', labelpad=5, fontsize=14)

            # Peak colorbars
            cbar7 = plt.colorbar(scatter_7)
            cbar7.set_label('Baz. ($^\\circ$)', fontsize=15)
            cbar7.set_ticks([-4, -2, 0, 2, 4])

            cbar8 = plt.colorbar(scatter_8)
            cbar8.set_label('Slow. (s/$^\\circ$)', fontsize=15)
            cbar8.set_ticks([1, 3, 5, 7, 9])

            ###

            for ax in [ax1, ax2, ax4, ax5, ax7, ax8]:
                ax.grid(which='major', linestyle='--', alpha=.75)
                ax.grid(which='minor', axis='x', linestyle='--', alpha=.65)

            set_locators(ax1, 'slow')
            set_locators(ax4, 'slow')
            set_locators(ax2, 'baz')
            set_locators(ax5, 'baz')

            ax1.set_ylim(2, 10)
            ax2.set_ylim(-25, 25)
            ax4.set_ylim(2, 10)
            ax5.set_ylim(-25, 25)

            ax1.set_ylabel('Slowness (s/$^\\circ$)')
            ax2.set_ylabel('Backazimuth ($^\\circ$)')
            # ax4.set_ylabel('Slowness (s/$^\\circ$)')
            # ax5.set_ylabel('Bazi ($^\\circ$)')
            ax8.set_ylabel('Coherence')
            ax7.set_xlabel('Time (s)')
            ax8.set_xlabel('Time (s)')
            for ax in [ax4, ax5]:
                ax.set_xticks([])
                ax.set_ylabel('')
            for ax in [ax1, ax2,ax4, ax5]:
                ax.set_xlabel('')

            ax7.set_yticks([])
            ax7.margins(x=0)
            ax8.margins(x=0)

            utc_dt=''.join(str(int(x)) for x in deets['Origin'])
            time_list=deets['Origin']
            formatted_time = f"Event origin: {int(time_list[0])} {int(time_list[1]):02d} {int(time_list[2]):02d} {int(time_list[3]):02d}:{int(time_list[4]):02d}"

            fig.text(0.2, .95, 'Grid #{}; {}'.format(grid_number,formatted_time),fontsize=16,color='Teal', ha='center', va='center')

            fig_name=py_figs+'picks_gridnum_{}_{}_{}.jpg'.format(grid_number,utc_dt,'II')

            if clicker_onoff:
                zoom_factory(ax4)
                # ph = panhandler(fig, button=1)
                sl_klicker = clicker(
                   ax4,markers=["+"], markersize=14,colors=['maroon'])
            ####
    print('----------DONE------------\n')
    # retrun_dict={"baz":}
    return sl_klicker,slow_grd,baz_grd,deets,grid_number,utc_dt,pick_folder,ax5,max_mean

def run_klicker_baz(ax):
    zoom_factory(ax)
    klicker = clicker(ax,markers=["x"], markersize=14,colors=['magenta'])
    return klicker
##
def json_converter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def use_klicker_save_scts(pick_folder,grid_number,utc_dt,slow_grd,slow_click,baz_grd,baz_click,deets,max_mean,bazoff):

    fig_name_=pick_folder+'picks_gridnum_{}_{}_{}.jpg'.format(grid_number,utc_dt,'II')
    ### extract times of max coherence for picked clicks
    print("Slowness min/max:")
    xf_pick_slow,xf_pick_slow_95 = extract_max_coher_clicks(slow_grd,slow_click)
    print("Backazimuth min/max:")
    xf_pick_baz,xf_pick_baz_95 = extract_max_coher_clicks(baz_grd,baz_click)
    if abs(xf_pick_slow[0][0] - xf_pick_baz[0][0]) > 1:
        raise ValueError(f"slow_pick and baz_pick are not within 1 sec of each other.")
    else:
        print(f"slow_pick and baz_pick diff: {abs(xf_pick_slow[0][0] - xf_pick_baz[0][0]):.2f} sec")

    # Write the extracted values (deets) to a new file in json
    data = {
        "event_time":utc_dt,
        "grid#":grid_number,
        "SRC_LAT": deets["Event"][0],
        "SRC_LON": deets["Event"][1],
        "SRC_DEP": deets["Event"][2],
        "REC_LAT": deets["ArrCen"][0],
        "REC_LON": deets["ArrCen"][1],
        "DIST": deets["Dist"][0],
        "BAZ": deets["Baz"][0],
        "max_mean":float(max_mean),
        "baz_offset":bazoff,
        "sct": [],
        "slow_click_raw":slow_click,
        "baz_click_raw":baz_click
        }

    for i, picks in enumerate(xf_pick_slow_95):

        data["sct"].append({
            "SCAT_time_max": float(xf_pick_slow[i][0]),
            "SCAT_slow_max": float(xf_pick_slow[i][1]),
            "SCAT_baz_max": float(xf_pick_baz[i][1]),
            "SCAT_slow_5_delta": float(picks[0]),
            "SCAT_sl_time_5_delta": float(picks[1]),
            "SCAT_baz_5_delta": float(xf_pick_baz_95[i][0]),
            "SCAT_bz_time_5_delta": float(xf_pick_baz_95[i][1]),
        })

    outfile=pick_folder+'grid_num_{}_{}_PICKS.json'.format(grid_number,utc_dt)

    with open(outfile, "w") as file:
        json.dump(data, file, indent=2,default=json_converter)
    fig_name=pick_folder+'picks_gridnum_{}_{}_picked.jpg'.format(grid_number,utc_dt)
    plt.savefig(fig_name,dpi=300,bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    plt.ion()
    plot_amp_factor=3
    folder_pattern = "sac_files/*"
    clicker_onoff=True
    grid_num=114 # choose this..if None, it runs for all grids in the folder.
    #STEP 1
    sl_klicker,slow_grd,baz_grd,deets,grid_number,utc_dt,pick_folder,ax_baz,max_mean=plot_vespa_pick_slow(folder_pattern,grid_num,clicker_onoff,plot_amp_factor)
    #when picking scatteres, the left click should be high slow/baz and right click low slow/baz!!!
    if not clicker_onoff:
        print('Exiting as clicker_onoff set as False')
        sys.exit()
    # print('RETURN TO KEEP GOING....')
    val1 = input("Click on slowness/time..press return when done!")

    #STEP2
    slow_click=sl_klicker.get_positions()
    klicker_baz=run_klicker_baz(ax_baz)

    # STEP 3
    val2 = input("choose on backazimuth/time.. enter BazOffset value when done and hit return: ")

    baz_click=klicker_baz.get_positions()
    use_klicker_save_scts(pick_folder,grid_number,utc_dt,slow_grd,slow_click,baz_grd,baz_click,deets,max_mean,val2)
    print('------------------------------------------------------------------')
    print('Picked scatters info saved as Json in sac_files/eq_folder/py_picks')
    ###

if __name__== "__main__":
    main()
