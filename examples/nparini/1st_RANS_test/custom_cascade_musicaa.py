import json
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import subprocess
import scipy.interpolate as si
import sys
import time

from aero_optim.simulator.simulator import WolfSimulator
from aero_optim.utils import (custom_input, find_closest_index, check_dir,
                              read_next_line_in_file, cp_filelist)

from typing import Callable

sys.path.insert(0, "/home/nparini/beta-aero-optim/examples/MultifidelityOptimization")
from cascade_adap.custom_cascade import CustomEvolution as WolfCustomEvolution # noqa
from cascade_adap.custom_cascade import CustomOptimizer as WolfCustomOptimizer # noqa
# from RANS_bruteForce.custom_cascade_wolf import CustomEvolution as WolfCustomEvolution # noqa
# from RANS_bruteForce.custom_cascade_wolf import CustomOptimizer as WolfCustomOptimizer # noqa

from aero_optim.optim.optimizer import WolfOptimizer
from pymoo.core.problem import Problem
from scipy.spatial.distance import cdist
from aero_optim.geom import (get_area, get_camber_th, get_chords, get_circle,
                             get_cog, split_profile, plot_profile, plot_sides)
from aero_optim.mesh.cascade_mesh import CascadeMesh
from aero_optim.optim.optimizer import WolfOptimizer

logger = logging.getLogger(__name__)

def get_feos_info(sim_outdir: str) -> dict:
    """
    **Reads** the feos_*.ini file from MUSICAA.
    """
    # regular expression to match lines with a name and a corresponding value
    pattern = re.compile(r"([A-Za-z\s]+)\.{2,}\s*(\S+)")

    # iterate over each line and apply the regex pattern
    feos_info: dict = {}
    with open(os.path.join(sim_outdir, "feos_air.ini"), "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                key = match.group(1).strip()
                value = match.group(2)
                feos_info[key] = float(value)
    return feos_info

def get_history(sim_outdir: str) -> dict:
    """
    **Reads** the history.dat file containing Mach, Pressure and error of completed 
    runs @ MP_inlet.
    """
    history_file = os.path.join(sim_outdir, "history.dat")
    history_info: dict = {}
    if not os.path.isfile(history_file):
        logger.warning("history.dat not found")
        return history_info
    data = np.loadtxt(history_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    history_info["n_partial"] = data[:, 0]
    history_info["mach"]      = data[:, 1]
    history_info["pressure"]  = data[:, 2]
    history_info["error"]     = data[:, 3]
    return history_info

def save_history(sim_outdir: str, history: dict):
    """
    **Writes** the history.dat file ontaining Mach, Pressure and error of completed 
    runs @ MP_inlet.
    """
    history_file = os.path.join(sim_outdir, "history.dat")
    with open(history_file, "a") as f:
        f.write(f"{history['n_partial']} {history['mach']} {history['pressure']} {history['error']}\n")

def get_mp1_data(sim_outdir: str) -> dict:
    """
    **Reads** the MP1_1.dat file from MUSICAA, containing data @ MP1
    used during simulation.
    """
    mp1_file = os.path.join(sim_outdir, "MP1_1.dat")
    if not os.path.isfile(mp1_file):
        logger.warning("MP1_1.dat not found")
        return {}
    data = np.loadtxt(mp1_file)
    mp1_info: dict = {}
    mp1_info["n_partial"] = data[-1, 0]
    mp1_info["beta"]      = data[-1, 1]
    mp1_info["mach"]      = data[-1, 2]
    mp1_info["mach_std"]  = data[-1, 3]
    mp1_info["pressure_tot"] = data[-1, 4]
    mp1_info["temperature_tot"] = data[-1, 5]
    mp1_info["pressure_back"]  = data[-1, 6]
    return mp1_info


def get_sim_info(sim_outdir: str):
    """
    **Returns** a dictionary containing relevant information on the mesh
    used by MUSICAA: number of blocks, block size, number of ghost points.

    Note: this element is automatically produced by MUSICAA, and need not be
          created manually.
    """
    # read relevant lines
    sim_info: dict = {}
    # info.ini creation may take time
    if not os.path.isfile(os.path.join(sim_outdir, 'info.ini')):
        logger.warning("info.ini not found")
        time.sleep(2.)
    with open(os.path.join(sim_outdir, "info.ini"), "r") as f:
        lines = f.readlines()
    try:
        sim_info["nbloc"] = int(lines[0].split()[4])
        for bl in range(sim_info["nbloc"]):
            bl += 1
            sim_info[f"block_{bl}"] = {}
            sim_info[f"block_{bl}"]["nx"] = int(lines[bl].split()[5])
            sim_info[f"block_{bl}"]["ny"] = int(lines[bl].split()[6])
            sim_info[f"block_{bl}"]["nz"] = int(lines[bl].split()[7])
        sim_info['Lref'] = float(lines[bl + 8].split()[-3])
        sim_info["ngh"] = int(lines[bl + 8].split()[-1])
        sim_info["dt"] = float(lines[bl + 9].split()[-1])
    except IndexError:
        logger.warning("info.ini incomplete")
        time.sleep(1.)
        return get_sim_info(sim_outdir)
    return sim_info


def get_time_info(sim_outdir: str) -> dict:
    """
    **Reads** the time.ini file from MUSICAA.
    """
    # regular expression to match lines with a name and a corresponding value
    pattern = re.compile(r"(\d{4}_\d{4})\s*=\s*(\d+)\s*([\d\.]+)\s*([\d\.E+-]+)")

    # iterate over each line and apply the regex pattern
    time_info: dict = {}
    with open(os.path.join(sim_outdir, "time.ini"), "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp = match.group(1)
                iter = int(match.group(2))
                cputot = float(match.group(3))
                time = float(match.group(4))

                # Store the extracted values in the dictionary
                time_info[timestamp] = {
                    'iter': iter,
                    'cputot': cputot,
                    'time': time
                }
        time_info["niter_total"] = iter
    return time_info


def get_niter_ftt(sim_outdir: str, L_ref: float) -> int:
    """
    **Returns** the number of iterations per flow-through time (ftt)
    """
    # f.t.t = L_ref/u_ref
    feos_info = get_feos_info(sim_outdir)
    Mach_ref = float(read_next_line_in_file("param.ini",
                                            "Reference Mach"))
    T_ref = float(read_next_line_in_file("param.ini",
                                         "Reference temperature"))
    c_ref = np.sqrt(feos_info["Equivalent gamma"] * feos_info["Gas constant"] * T_ref)
    u_ref = c_ref * Mach_ref
    Lgrid = float(read_next_line_in_file("param.ini",
                                         "Scaling value for the grid Lgrid"))
    if Lgrid != 0:
        L_ref = L_ref * Lgrid
    ftt = L_ref / u_ref
    sim_info = get_sim_info(sim_outdir)
    return int(ftt / sim_info["dt"])


def read_bl(sim_outdir: str, bl: int) -> tuple[np.ndarray, np.ndarray]:
    """
    **Reads** simulation grid coordinates.
    """
    # get sim_info
    sim_info = get_sim_info(sim_outdir)
    nx = sim_info[f"block_{bl}"]["nx"]
    ny = sim_info[f"block_{bl}"]["ny"]
    ngh = sim_info["ngh"]
    nx_ext = nx + 2 * ngh
    ny_ext = ny + 2 * ngh

    # read coordinates extended by ghost cells
    filename = os.path.join(sim_outdir, f"grid_bl{bl}_ngh{ngh}.bin")
    f = open(filename, "r")
    x = np.fromfile(f, dtype=("<f8"), count=nx_ext * ny_ext).reshape((nx_ext, ny_ext),
                                                                     order="F")
    y = np.fromfile(f, dtype=("<f8"), count=nx_ext * ny_ext).reshape((nx_ext, ny_ext),
                                                                     order="F")
    x = x[ngh:-ngh, ngh:-ngh]
    y = y[ngh:-ngh, ngh:-ngh]
    return x, y


def mixed_out(data: dict) -> dict:
    """
    **Computes** mixed-out quantities:
    see A. Prasad (2004): https://doi.org/10.1115/1.1928289
    """
    # conservation of mass
    m_bar = np.nanmean(data["rhou_interp"])
    v_bar = np.nanmean(data["rho*uv_interp"]) / m_bar
    w_bar = np.nanmean(data["rho*uw_interp"]) / m_bar
    vv_bar = v_bar**2
    ww_bar = w_bar**2

    # conservation of momentum
    x_mom = np.nanmean(data["rho*uu_interp"] + data["p_interp"])
    y_mom = np.nanmean(data["rho*uv_interp"])
    z_mom = np.nanmean(data["rho*uw_interp"])

    # conservation of energy
    gam = data["gam"]
    R = data["R"]
    e = data["R"] * data["gam"] / (data["gam"] - 1) *\
        np.nanmean(data["rhou_interp"] * data["T_interp"]) +\
        0.5 * np.nanmean(data["rhou_interp"] * (data["uu_interp"]
                                                + data["vv_interp"]
                                                + data["ww_interp"]))

    # quadratic equation
    Q = 1 / m_bar**2 * (1 - 2 * gam / (gam - 1))
    L = 2 / m_bar**2 * (gam / (gam - 1) * x_mom - x_mom)
    C = 1 / m_bar**2 * (x_mom**2 + y_mom**2 + z_mom**2) - 2 * e / m_bar

    # select subsonic root
    p_bar = (-L - np.sqrt(L**2 - 4 * Q * C)) / 2 / Q
    u_bar = (x_mom - p_bar) / m_bar
    V2_bar = u_bar**2 + vv_bar + ww_bar
    rho_bar = m_bar / u_bar
    T_bar = p_bar / rho_bar / R
    c_bar = np.sqrt(gam * R * T_bar)
    M_bar = np.sqrt(V2_bar) / c_bar
    p0_bar = p_bar * (1 + (gam - 1) / 2 * M_bar**2)**(gam / (gam - 1))

    # store
    mixed_out_state = {"p_bar": p_bar,
                       "rho_bar": rho_bar,
                       "T_bar": T_bar,
                       "V2_bar": V2_bar,
                       "M_bar": M_bar,
                       "p0_bar": p0_bar}

    return mixed_out_state


def read_stats_bl(sim_outdir: str, bl_list: list[int], var_list: list[str]) -> dict:
    """
    **Reads** statistics of MUSICAA computation block from stats*_bl*.bin.
    """
    # get simulation information
    sim_info = get_sim_info(sim_outdir)

    # list of variables in file
    vars1 = ["rho", "u", "v", "w", "p", "T", "rhou", "rhov", "rhow", "rhoe",
             "rho**2", "uu", "vv", "ww", "uv", "uw", "vw", "vT", "p**2", "T**2",
             "mu", "divloc", "divloc**2"]
    vars2 = ["e", "h", "c", "s", "M", "0.5*q", "g", "la", "cp", "cv",
             "prr", "eck", "rho*dux", "rho*duy", "rho*duz", "rho*dvx", "rho*dvy",
             "rho*dvz", "rho*dwx", "rho*dwy", "rho*dwz", "p*div", "rho*div", "b1",
             "b2", "b3", "rhoT", "uT", "vT", "e**2", "h**2", "c**2", "s**2",
             "qq/cc2", "g**2", "mu**2", "la**2", "cv**2", "cp**2", "prr**2", "eck**2",
             "p*u", "p*v", "s*u", "s*v", "p*rho", "h*rho", "T*p", "p*s", "T*s", "rho*s",
             "g*rho", "g*p", "g*s", "g*T", "g*u", "g*v", "p*dux", "p*dvy", "p*dwz",
             "p*duy", "p*dvx", "rho*div**2", "dux**2", "duy**2", "duz**2", "dvx**2",
             "dvy**2", "dvz**2", "dwx**2", "dwy**2", "dwz**2", "b1**2", "b2**2", "b3**2",
             "rho*b1", "rho*b2", "rho*b3", "rho*uu", "rho*vv", "rho*ww",
             "rho*T**2", "rho*b1**2", "rho*b2**2", "rho*b3**2", "rho*uv", "rho*uw",
             "rho*vw", "rho*vT", "rho*u**2*v", "rho*v**3", "rho*w**2*v", "rho*v**2*u",
             "rho*dux**2", "rho*dvy**2", "rho*dwz**2", "rho*duy*dvx", "rho*duz*dwx",
             "rho*dvz*dwy", "u**3", "p**3", "u**4", "p**4", "Frhou", "Frhov", "Frhow",
             "Grhov", "Grhow", "Hrhow", "Frhovu", "Frhouu", "Frhovv", "Frhoww",
             "Grhovu", "Grhovv", "Grhoww", "Frhou_dux", "Frhou_dvx", "Frhov_dux",
             "Frhov_duy", "Frhov_dvx", "Frhov_dvy", "Frhow_duz", "Frhow_dvz",
             "Frhow_dwx", "Grhov_duy", "Grhov_dvy", "Grhow_duz", "Grhow_dvz",
             "Grhow_dwy", "Hrhow_dwz", "la*dTx", "la*dTy", "la*dTz",
             "h*u", "h*v", "h*w", "rho*h*u", "rho*h*v", "rho*h*w", "rho*u**3",
             "rho*v**3", "rho*w**3", "rho*w**2*u",
             "h0", "e0", "s0", "T0", "p0", "rho0", "mut"]
    all_vars = [vars1, vars2]

    # loop over blocks
    data: dict = {}
    stats = 1
    for vars in all_vars:
        for bl in bl_list:
            # get filename
            filename = os.path.join(sim_outdir, f"stats{stats}_bl{bl}.bin")

            # get block dimensions
            nx = sim_info[f"block_{bl}"]["nx"]
            ny = sim_info[f"block_{bl}"]["ny"]

            # read and store
            if stats == 1:
                data[f"block_{bl}"] = {}
            logger.info(f"wait for {filename} to be produced..")
            while not os.path.isfile(filename):
                time.sleep(1.)
            logger.info(f"{filename} found")
            f = open(filename, "rb")
            dtype = np.dtype("f8")
            for var in vars:
                data[f"block_{bl}"][var] = np.fromfile(
                    f, dtype=dtype, count=nx * ny).reshape((nx, ny), order="F")
            f.close()

            # keep only those provided in var_list
            unwanted = set(data[f"block_{bl}"]) - set(var_list)
            for unwanted_key in unwanted:
                if unwanted_key != "x" and unwanted_key != "y":
                    del data[f"block_{bl}"][unwanted_key]

        stats += 1

    # add fluid properties
    feos_info = get_feos_info(sim_outdir)
    data["gam"] = feos_info["Equivalent gamma"]
    data["R"] = feos_info["Gas constant"]

    return data


def extract_measurement_line(sim_outdir: str,
                             bl_list: list[int],
                             lims: list[float]) -> dict:
    """
    **Extracts** data along measurement line.
    """
    # get time-averaged block data
    var_list_interp = ["uu", "vv", "ww", "rhou", "rhov", "rho*uu", "rho*uv", "rho*uw", "p", "T"]
    data = read_stats_bl(sim_outdir, bl_list, var_list_interp)

    # get coordinates
    for bl in bl_list:
        data[f"block_{bl}"]["x"], data[f"block_{bl}"]["y"] = read_bl(sim_outdir, bl)

    # interpolate all variables
    for var in var_list_interp:
        data[f"{var}_interp"] = line_interp(data, var, lims, bl_list)

    return data


def line_interp(data: dict, var: str, lims: list[float], bl_list: list[int]) -> np.ndarray:
    """
    **Interpolates** data along a line defined by lims.
    """
    # flatten data
    var_flat_ = []
    x_flat_ = []
    y_flat_ = []
    for bl in bl_list:
        var_flat_.append(data[f"block_{bl}"][f"{var}"].flatten())
        x_flat_.append(data[f"block_{bl}"]["x"].flatten())
        y_flat_.append(data[f"block_{bl}"]["y"].flatten())
    var_flat = np.hstack(var_flat_)
    x_flat = np.hstack(x_flat_)
    y_flat = np.hstack(y_flat_)

    # create line
    x1, y1 = lims[0], lims[1]
    x2, y2 = lims[2], lims[3]
    x_interp = np.linspace(x1, x2, 1000)
    y_interp = np.linspace(y1, y2, 1000)

    # interpolate
    var_interp = si.griddata((x_flat, y_flat), var_flat,
                             (x_interp, y_interp), method="linear")

    return var_interp


def compute_QoIs(config: dict, sim_outdir: str) -> pd.DataFrame:
    """
    **Returns** the QoIs during computation in a DataFrame.
    """
    qty_list: list[list[float]] = []
    head_list: list[str] = []
    post_process_args: dict = config["simulator"]["post_process"]
    # loop over the post-processing arguments to extract from the results
    for qty in post_process_args["outputs"]:
        # check if the method for computing qty exists
        try:
            # get arguments
            get_args: Callable = globals()[f"args_{qty}"]
            args = get_args(sim_outdir, config)
            get_value: Callable = globals()[qty]
            value = get_value(sim_outdir, args)
        except AttributeError:
            raise Exception(f"ERROR -- method for computing {qty} does not exist")
        try:
            # compute simulation results
            qty_list.append(value)
            head_list.append(qty)
        except Exception as e:
            logger.warning(f"could not compute {qty}")
            logger.warning(f"exception {e} was raised")
    # pd.Series allows columns of different lengths
    df = pd.DataFrame({head_list[i]: pd.Series(qty_list[i]) for i in range(len(qty_list))})
    return df

def read_fortran_record(f, dtype, count=1, shape=None):
    """
    **Reads** a Fortran unformatted record (MUSICAA postprocess results).
    """
    f.read(4)  # leading marker
    if dtype == 'int':
        data = np.frombuffer(f.read(4 * count), dtype='>i4')
    elif dtype == 'float':
        data = np.frombuffer(f.read(4 * count), dtype='>f4')
    elif dtype == 'double':
        data = np.frombuffer(f.read(8 * count), dtype='>f8')
    f.read(4)  # trailoutg marker
    if shape is not None:
        return data.reshape(shape, order='F')  
    return int(data[0]) if count == 1 and dtype == 'int' else data[0] if count == 1 else data

def read_ml(filepath):
    """
    **Reads** ml_in.bin or ml_out.bin .
    """
    with open(filepath, 'rb') as f:
        nx_ml  = read_fortran_record(f, 'int')
        ny_ml  = read_fortran_record(f, 'int')
        n      = nx_ml * ny_ml
        shape  = (nx_ml, ny_ml)

        x_ml   = read_fortran_record(f, 'double', n, shape)
        y_ml   = read_fortran_record(f, 'double', n, shape)

        # 12 variables: <rho>,<u>,<v>,<w>,<p>,<T>,<u'u'>,<v'v'>,<w'w'>,<u'v'>,<mu>,<M>
        rho_ml = read_fortran_record(f, 'double', n, shape)
        u_ml   = read_fortran_record(f, 'double', n, shape)
        v_ml   = read_fortran_record(f, 'double', n, shape)
        w_ml   = read_fortran_record(f, 'double', n, shape)
        p_ml   = read_fortran_record(f, 'double', n, shape)
        T_ml   = read_fortran_record(f, 'double', n, shape)
        uu_ml  = read_fortran_record(f, 'double', n, shape)
        vv_ml  = read_fortran_record(f, 'double', n, shape)
        ww_ml  = read_fortran_record(f, 'double', n, shape)
        uv_ml  = read_fortran_record(f, 'double', n, shape)
        mu_ml  = read_fortran_record(f, 'double', n, shape)
        M_ml   = read_fortran_record(f, 'double', n, shape)

    # derived
    W_ml    = np.sqrt(u_ml**2 + v_ml**2)
    beta_ml = np.arctan2(v_ml, u_ml) * 180 / np.pi + 90

    return dict(nx=nx_ml, ny=ny_ml, x=x_ml, y=y_ml,
                rho=rho_ml, u=u_ml, v=v_ml, w=w_ml,
                p=p_ml, T=T_ml, uu=uu_ml, vv=vv_ml, ww=ww_ml,
                uv=uv_ml, mu=mu_ml, M=M_ml,
                W=W_ml, beta=beta_ml)


def read_mixed_out(filepath, nx_ml):
    """
    **Reads** mixed_out_ml_in.bin or mixed_out_ml_out.bin .
    """
    with open(filepath, 'rb') as f:
        nxml  = read_fortran_record(f, 'int')
        xml   = read_fortran_record(f, 'double', nx_ml)

        # Prasad
        pmo_P = read_fortran_record(f, 'double', nx_ml)

        # Bloch
        pmo_B = read_fortran_record(f, 'double', nx_ml)
        Wmo_B = read_fortran_record(f, 'double', nx_ml)
        bmo_B = read_fortran_record(f, 'double', nx_ml)
        rmo_B = read_fortran_record(f, 'double', nx_ml)

        # Schreiber & Starken
        pmo_S = read_fortran_record(f, 'double', nx_ml)
        Wmo_S = read_fortran_record(f, 'double', nx_ml)
        bmo_S = read_fortran_record(f, 'double', nx_ml)
        rmo_S = read_fortran_record(f, 'double', nx_ml)
        Mmo_S = read_fortran_record(f, 'double', nx_ml)

    return dict(nxml=nxml, xml=xml,
                pmo_P=pmo_P,
                pmo_B=pmo_B, Wmo_B=Wmo_B, bmo_B=bmo_B, rmo_B=rmo_B,
                pmo_S=pmo_S, Wmo_S=Wmo_S, bmo_S=bmo_S, rmo_S=rmo_S, Mmo_S=Mmo_S)


def read_ml_and_mixed_out(rep, ind):
    """
    ind=1 -> inlet  (ml_in.bin  + mixed_out_ml_in.bin)
    ind=2 -> outlet (ml_out.bin + mixed_out_ml_out.bin)
    """
    assert ind in (1, 2), "ind must be 1 (inlet) or 2 (outlet)"

    ml_file      = rep + ('/ml_in.bin'           if ind == 1 else '/ml_out.bin')
    mixed_file   = rep + ('/mixed_out_ml_in.bin'  if ind == 1 else '/mixed_out_ml_out.bin')

    ml   = read_ml(ml_file)
    mo   = read_mixed_out(mixed_file, ml['nx'])

    return ml, mo

def args_MixedoutOmega(sim_outdir: str, config: dict) -> dict:
    """
    **Returns** a dictionary containing the required arguments for MixedoutOmega.
    """
    args: dict = {}
    ml_in,  mo_in  = read_ml_and_mixed_out(sim_outdir, ind=1)
    ml_out, mo_out = read_ml_and_mixed_out(sim_outdir, ind=2)
    info = get_sim_info(sim_outdir)
    mp_in = config["simulator"]["post_process"]["measurement_lines"]["inlet_x1"]
    #
    Lscale = info['Lref']
    gam=1.4
    gam1=gam-1
    pi=3.1415
    mp1_idx = np.argmin(np.abs(np.abs(ml_in['x'][:,1]/ Lscale)  - mp_in))
    #
    args['P1'] = np.mean(ml_in['p'][mp1_idx,:])
    args['T1'] = np.mean(ml_in['T'][mp1_idx,:])
    args['M1'] = np.mean(ml_in['M'][mp1_idx,:])
    args['U1'] = np.mean(ml_in['u'][mp1_idx,:])
    args['V1'] = np.mean(ml_in['v'][mp1_idx,:])
    args['beta1']=np.arctan(args['V1']/args['U1'])*180/pi+90
    cc1 = (1 + (gam1 / 2) * args['M1']**2)**(gam / gam1)

    args['P2'] = np.mean(mo_out['pmo_B'])
    args['M2'] = np.mean(mo_out['Mmo_S'])
    cc2 = (1 + (gam1 / 2) * args['M2']**2)**(gam / gam1)

    args['P1_tot'] = args['P1'] * cc1
    args['P2_tot'] = args['P2'] * cc2
    args['pressure_ratio'] = args['P2'] / args['P1']

    return args

def MixedoutOmega(sim_outdir: str, args: dict) -> float:
    """
    **Post-processes** the results of a terminated simulation, using MUSICAA 
    postprocess results.
    **Returns** the extracted results in a DataFrame.
    """
    return (args['P1_tot'] - args['P2_tot']) /\
         (args['P1_tot'] - args['P1'])



def args_MixedoutLossCoef(sim_outdir: str, config: dict) -> dict:
    """
    **Returns** a dictionary containing the required arguments for MixedoutLossCoef.
    """
    args: dict = {}

    # inlet
    bl_list = config["gmsh"]["inlet_bl"]
    x1 = config["simulator"]["post_process"]["measurement_lines"]["inlet_x1"]
    x2 = config["simulator"]["post_process"]["measurement_lines"]["inlet_x2"]
    x, y = read_bl(sim_outdir, bl_list[0])
    closest_index = find_closest_index(x[:, 0], x1)
    y1 = y[closest_index, :].min()
    y2 = y1 + config["gmsh"]["pitch"]
    inlet_lims = [x1, y1, x2, y2]
    args["inlet_bl"] = bl_list
    args["inlet_lims"] = inlet_lims

    # outlet
    bl_list = config["gmsh"]["outlet_bl"]
    x1 = config["simulator"]["post_process"]["measurement_lines"]["outlet_x1"]
    x2 = config["simulator"]["post_process"]["measurement_lines"]["outlet_x2"]
    x, y = read_bl(sim_outdir, bl_list[0])
    closest_index = find_closest_index(x[:, 0], x1)
    y1 = y[closest_index, :].min()
    y2 = y1 + config["gmsh"]["pitch"]
    outlet_lims = [x1, y1, x2, y2]
    args["outlet_bl"] = bl_list
    args["outlet_lims"] = outlet_lims

    return args


def MixedoutLossCoef(sim_outdir: str, args: dict) -> float:
    """
    **Post-processes** the results of a terminated simulation.
    **Returns** the extracted results in a DataFrame.
    """
    # extract arguments
    inlet_bl = args["inlet_bl"]
    inlet_lims = args["inlet_lims"]
    outlet_bl = args["outlet_bl"]
    outlet_lims = args["outlet_lims"]

    # compute inlet mixed-out pressure
    inlet_data = extract_measurement_line(sim_outdir, inlet_bl, inlet_lims)
    inlet_mixed_out_state = mixed_out(inlet_data)

    # compute oulet mixed-out pressure
    outlet_data = extract_measurement_line(sim_outdir, outlet_bl, outlet_lims)
    outlet_mixed_out_state = mixed_out(outlet_data)

    return (inlet_mixed_out_state["p0_bar"] - outlet_mixed_out_state["p0_bar"]) /\
           (inlet_mixed_out_state["p0_bar"] - inlet_mixed_out_state["p_bar"])


def args_OutflowAngle(sim_outdir: str, config: dict) -> dict:
    return args_MixedoutLossCoef(sim_outdir, config)


def OutflowAngle(sim_outdir: str, args: dict) -> float:
    """
    **Post-processes** the results of a terminated simulation.
    **Returns** the extracted results in a DataFrame.
    """
    # extract arguments
    outlet_bl = args["outlet_bl"]
    outlet_lims = args["outlet_lims"]

    # compute oulet mixed-out pressure
    outlet_data = extract_measurement_line(sim_outdir, outlet_bl, outlet_lims)
    outflow_angle = np.nanmean(np.arctan(outlet_data["rhov_interp"] / outlet_data["rhou_interp"]))

    return outflow_angle / np.pi * 180


# class CustomSimulator(Simulator):
class CustomSimulator(WolfSimulator):
    """
    This class implements a simulator for the CFD code MUSICAA.
    """
    def process_config(self):
        """
        **Makes sure** the config file contains the required information and extracts it:

        - computation_type (str): type of computation (steady/unsteady)
        """
        logger.debug("processing config..")
        if "exec_cmd" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <exec_cmd> entry in {self.config['simulator']}")
        if "ref_input" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <ref_input> entry in {self.config['simulator']}")
        if "post_process" not in self.config["simulator"]:
            logger.debug(f"no <post_process> entry in {self.config['simulator']}")
        self.computation_type = read_next_line_in_file("param.ini", "DES without subgrid")
        self.computation_type = "unsteady" if self.computation_type == "N" else "steady"

    def set_solver_name(self):
        """
        **Sets** the solver name to musicaa.
        """
        self.solver_name = "musicaa"

    def pre_process(self, meshfile: str, gid: int, cid: int) -> tuple[str, list[str]]:
        """
        **Pre-processes** the simulation execution
        and **returns** the execution command and directory.
        """
        # get the simulation meshfile
        if meshfile:
            full_meshfile = meshfile
            path_to_meshfile = "/".join(full_meshfile.split("/")[:-1])
            meshfile = full_meshfile.split("/")[-1]
        else:
            path_to_meshfile = self.config["gmsh"]["mesh_dir"]
            meshfile = self.config["gmsh"]["mesh_name"]

        # name of simulation directory
        sim_outdir = self.get_sim_outdir(gid=gid, cid=cid)
        check_dir(sim_outdir)

        # copy files and executable to directory
        cp_filelist(self.config["simulator"]["cp_list"],
                    [sim_outdir] * len(self.config["simulator"]["cp_list"]))
        logger.info((f"param.ini and "
                     f"param_blocks.ini "
                     f"copied to {sim_outdir}"))

        # modify solver input file: delete half-cell at block boundaries
        args: dict = {}
        args.update({"from_interp": "0"})
        args.update({"Max number of temporal iterations": "500000 3000.0"})
        args.update({"Iteration number to start statistics": "9999999"})
        args.update({"Half-cell": "T", "Coarse grid": "F 0", "Perturb grid": "F"})
        args.update({"Directory for grid files":
                     f"'{os.path.relpath(path_to_meshfile, sim_outdir)}'"})
        args.update({"Name for grid files": meshfile})
        if self.computation_type == "steady":
            args.update({"Compute residuals": "T"})
        custom_input(os.path.join(sim_outdir, "param.ini"), args)

        # execute MUSICAA to delete half-cell
        os.chdir(sim_outdir)
        preprocess_cmd = self.config["simulator"]["preprocess_cmd"].split()
        with open(f"{self.solver_name}_g{gid}_c{cid}_half-cell.out", "wb") as out:
            with open(f"{self.solver_name}_g{gid}_c{cid}_half-cell.err", "wb") as err:
                logger.info(f"delete mesh half-cell for g{gid}, c{cid} with {self.solver_name}")
                subprocess.run(preprocess_cmd,
                               env=os.environ,
                               stdin=subprocess.DEVNULL,
                               stdout=out,
                               stderr=err,
                               universal_newlines=True)

        # modify solver input file: mode from_scratch
        args.update({"from_interp": "1"})
        custom_input("param.ini", args)
        logger.info(f"changed execution mode to 1 in {sim_outdir}")
        os.chdir(self.cwd)

        # create local config file
        sim_config = {
            "gmsh": self.config["gmsh"],
            "optim": self.config["optim"],
            "simulator": self.config["simulator"],
        }
        with open(os.path.join(sim_outdir, "sim_config.json"), "w") as jfile:
            json.dump(sim_config, jfile)

        return sim_outdir, self.exec_cmd

    def _post_process(self, dict_id: dict, sim_outdir: str) -> str:
        """
        **Post-processes** the results of a terminated simulation.
        **Returns** the extracted results in a DataFrame.
        """
        df = pd.read_csv(os.path.join(sim_outdir, self.config["simulator"]["post_process"]["file"]))
        logger.info(
            f"g{dict_id['gid']}, c{dict_id['cid']} converged in {len(df)} ftt."
        )
        logger.info(f"last values:\n{df.tail(n=1).to_string(index=False)}")
        return df

    def post_process(self, dict_id: dict, sim_out_dir: str) -> dict[str, pd.DataFrame]:
        """
        **Post-processes** the results of a terminated triple simulation.</br>
        **Returns** the extracted results in a dictionary of DataFrames.

        Note:
            there are two oIoIs: loss_ADP and loss_OP = 1/2(loss_OP1 + loss_OP2)
            to be extracted from: sim_out_dir/ADP, sim_out_dir/OP1  and sim_out_dir/OP2
        """
        df_sub_dict: dict[str, pd.DataFrame] = {}
        for fname in ["ADP", "OP1", "OP2"]:
            logger.debug(f"post_process g{dict_id['gid']}, c{dict_id['cid']} {fname}..")
            df_sub_dict[fname] = self._post_process(dict_id, os.path.join(sim_out_dir, fname))
        return df_sub_dict


# class CustomOptimizer(WolfCustomOptimizer):
#     def _observe(self, pop_fitness: np.ndarray):
#         """
#         **Plots** some results each time a generation has been evaluated:</br>
#         > the simulations residuals,</br>
#         > the candidates fitnesses,</br>
#         > the baseline and deformed profiles.
#         """
#         gid = self.gen_ctr
#
#         # plot settings
#         baseline: np.ndarray = self.ffd.pts
#         profiles: list[np.ndarray] = self.ffd_profiles[gid]
#         res_dict = self.simulator.df_dict[gid]
#         df_key = res_dict[self.feasible_cid[gid][0]]["ADP"].columns  # ResTot, LossCoef, x, y, Mis
#         cmap = mpl.colormaps[self.cmap].resampled(self.doe_size)
#         colors = cmap(np.linspace(0, 1, self.doe_size))
#         # subplot construction
#         fig = plt.figure(figsize=(16, 16))
#         ax1 = plt.subplot(2, 1, 1)  # profiles
#         ax2 = plt.subplot(2, 3, 4)  # loss_ADP
#         ax3 = plt.subplot(2, 3, 5)  # loss_OP
#         ax4 = plt.subplot(2, 3, 6)  # fitness (loss_ADP vs loss_OP)
#         plt.subplots_adjust(wspace=0.25)
#         ax1.plot(baseline[:, 0], baseline[:, 1], color="k", lw=2, ls="--", label="baseline")
#         # loop over candidates through the last generated profiles
#         for cid in self.feasible_cid[gid]:
#             ax1.plot(profiles[cid][:, 0], profiles[cid][:, 1], color=colors[cid], label=f"c{cid}")
#             res_dict[cid]["ADP"][df_key[0]].plot(ax=ax2, color=colors[cid], label=f"c{cid}")
#             vsize = min(len(res_dict[cid]["OP1"][df_key[0]]), len(res_dict[cid]["OP2"][df_key[0]]))
#             ax3.plot(
#                 range(vsize),
#                 0.5 * (res_dict[cid]["OP1"][df_key[0]].values[-vsize:]
#                        + res_dict[cid]["OP2"][df_key[0]].values[-vsize:]),
#                 color=colors[cid],
#                 label=f"c{cid}"
#             )
#             ax4.scatter(pop_fitness[cid, 0], pop_fitness[cid, 1],
#                         color=colors[cid], label=f"c{cid}")
#         ax4.scatter(self.bsl_w_ADP, self.bsl_w_OP, marker="*", color="red", label="baseline")
#         # legend and title
#         fig.suptitle(
#             f"Generation {gid} results", size="x-large", weight="bold", y=0.93
#         )
#         # top
#         ax1.set_title("FFD profiles", weight="bold")
#         ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#         ax1.set_xlabel('x')
#         ax1.set_ylabel('y')
#         # bottom left
#         ax2.set_title(f"{df_key[0]} ADP", weight="bold")
#         ax2.set_xlabel('it. #')
#         ax2.set_ylabel('$w_\\text{ADP}$')
#         # bottom center
#         ax3.set_title(f"{df_key[0]} OP", weight="bold")
#         ax3.set_xlabel('it. #')
#         ax3.set_ylabel('$w_\\text{OP}$')
#         # bottom right
#         ax4.set_title(f"{self.QoI} ADP vs {self.QoI} OP", weight="bold")
#         ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#         ax4.set_xlabel('$w_\\text{ADP}$')
#         ax4.set_ylabel('$w_\\text{OP}$')
#         # save figure as png
#         fig_name = f"pymoo_g{gid}.png"
#         logger.info(f"saving {fig_name} to {self.figdir}")
#         plt.savefig(os.path.join(self.figdir, fig_name), bbox_inches='tight')
#         plt.close()
#
#
# class CustomEvolution(WolfCustomEvolution):
#     """Same custom class as the one defined in cascade_adap"""


def get_valid_center(
        x: np.ndarray, y: np.ndarray, dmin: float, dmax: float,
        le: bool = True, percent: float = 10, resolution: int = 50
) -> np.ndarray | None:
    """
    **Computes** and **returns** the center of a valid circle in regards of the
    leading/trailing edge constraints.

    In particular, it checks if circles of radius dmin can fit in both the leading
    and trailing edges, and if such circles have their centers located at a distance
    to the leading/trailing edge that is smaller than dmax.
    **Returns** None if the constraint is not respected.
    """
    # sort coordinates
    count = int(len(x) * percent / 100)
    indices = np.argsort(x) if le else np.argsort(x)[::-1]
    x_sorted = x[indices][:count]
    y_sorted = y[indices][:count]

    # find bounding box for circle center location search
    profile = np.column_stack([x_sorted, y_sorted])
    min_x, min_y = profile.min(axis=0)
    max_x, max_y = profile.max(axis=0)

    # generate grid of candidate points, starting near leading edge
    x_vals = np.linspace(min_x, max_x, resolution)
    y_vals = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    candidate_points = np.vstack([X.ravel(), Y.ravel()]).T

    # sort candidates by smallest x and on the right side of the le/te edge
    candidate_points = candidate_points[np.argsort(candidate_points[:, 0])]
    candidate_points = (
        candidate_points[candidate_points[:, 0] > profile[0, 0]] if le
        else candidate_points[candidate_points[:, 0] < profile[0, 0]]
    )

    # keep only the candidates at a distance from the le/te edge
    # comprised between dmin and dmax
    dists = cdist(candidate_points, np.array([[x_sorted[0], y_sorted[0]]]))
    idx, _ = np.where((dists > dmin) & (dists < dmax))

    # check if there is at least one pt that gives a valid circle center
    for pt in candidate_points[idx]:
        if np.min(cdist([pt], profile)) > dmin:
            return pt
    return None

class CustomOptimizer(WolfCustomOptimizer):
    """
    Same custom class as the one defined in cascade_adap.

    The optimization strategy consists of an updated version of the original cascade optimizer
    with a more complete set of constraints:
    - the leading/trailing edge radius constraints are enhanced and adapted to
      the LES mesh based profile,
    - constraints related to the outflow angle are also introduced.
    """
    def __init__(self, config: dict):
        """
         **Inner**

        - feasible_cid (dict[int, list[int]]): dictionary containing feasible cid of each gid.
        """
        WolfOptimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=2, n_ieq_constr=9, xl=self.bound[0], xu=self.bound[1]
        )
        self.feasible_cid: dict[int, list[int]] = {}

    def set_inner(self):
        """
        **Sets** some baseline quantities required to compute the relative constraints:

        - angle_ADP (list[float]): min/max outflow angle accepted deviations at ADP
        - angle_OP1 (list[float]): min/max outflow angle accepted deviations at OP1
        - angle_OP2 (list[float]): min/max outflow angle accepted deviations at OP2
        """
        super().set_inner()
        self.CoI = self.config["optim"].get("CoI", "OutflowAngle")
        self.angle_ADP = self.config["optim"].get("angle_ADP")
        self.angle_OP1 = self.config["optim"].get("angle_OP1")
        self.angle_OP2 = self.config["optim"].get("angle_OP2")

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Computes** the objective function and constraints for each candidate in the generation.

        Note:
            for this use-case, some of the constraints can be computed before simulations.
            Unfeasible candidates are not simulated.
        """

        ffff = self.config["study"]["outdir"].split("/")[-1]
        gid = self.gen_ctr
        self.feasible_cid[gid] = []
        with open(f'{self.outdir}/{ffff}_g{gid}.txt','w+') as fout:
            for ind in X:
                for var in ind:
                    fout.write(f"{var} ")
                fout.write(f"\n")

        # compute candidates geometric constraints and execute feasible candidates only
        geom_constraints = self.execute_constrained_candidates(X, gid)

        # update candidates fitness
        # Note: this time only the first value in the dataframe should be read

        for cid in range(len(X)):
            if cid in self.feasible_cid[gid]:
                loss_ADP = self.simulator.df_dict[gid][cid]["ADP"][self.QoI].dropna().iloc[-1]
                loss_OP1 = self.simulator.df_dict[gid][cid]["OP1"][self.QoI].dropna().iloc[-1]
                loss_OP2 = self.simulator.df_dict[gid][cid]["OP2"][self.QoI].dropna().iloc[-1]
                logger.info(f"g{gid}, c{cid}: "
                            f"w_ADP = {loss_ADP}, w_OP = {0.5 * (loss_OP1 + loss_OP2)}")
                self.J.append([loss_ADP, 0.5 * (loss_OP1 + loss_OP2)])
            else:
                self.J.append([float("nan"), float("nan")])

        # compute candidates angle constraints
        if not self.constraint:
            angle_constraints = [[-1.] * 3 for _ in range(len(X))]
        else:
            angle_constraints = []
            CoI = self.CoI
            for cid in range(len(X)):
                # print(f"g{gid}, c{cid}: Outflow angle constraints computation")
                # # Check if the directory exists
                # wolf_dir = os.path.join("output", "WOLF", f"wolf_g{gid}_c{cid}")
                # if not os.path.exists(wolf_dir):
                #     outflow_angle_ADP = 1000
                #     outflow_angle_OP1 = 1000
                #     outflow_angle_OP2 = 1000
                # else:
                outflow_angle_ADP = self.simulator.df_dict[gid][cid]["ADP"][CoI].dropna().iloc[-1]
                outflow_angle_OP1 = self.simulator.df_dict[gid][cid]["OP1"][CoI].dropna().iloc[-1]
                outflow_angle_OP2 = self.simulator.df_dict[gid][cid]["OP2"][CoI].dropna().iloc[-1]
                print(f"g{gid}, c{cid}: Outflow angles - ADP: {outflow_angle_ADP}, OP1: {outflow_angle_OP1}, OP2: {outflow_angle_OP2}")
                angle_constraints.append(
                    [self.angle_ADP[0] - outflow_angle_ADP if outflow_angle_ADP < self.angle_ADP[0]
                     else outflow_angle_ADP - self.angle_ADP[1],
                     self.angle_OP1[0] - outflow_angle_OP1 if outflow_angle_OP1 < self.angle_OP1[0]
                     else outflow_angle_OP1 - self.angle_OP1[1],
                     self.angle_OP2[0] - outflow_angle_OP2 if outflow_angle_OP2 < self.angle_OP2[0]
                     else outflow_angle_OP2 - self.angle_OP2[1]]
                )
                logger.debug(f"g{gid}, c{cid} ADP outflow angle: ({outflow_angle_ADP})")
                if angle_constraints[-1][0] > 0:
                    logger.info(f"g{gid}, c{cid} ADP outflow angle: constraint violation, should be between {self.angle_ADP[0]} and {self.angle_ADP[1]}")
                logger.debug(f"g{gid}, c{cid} OP1 outflow angle: ({outflow_angle_OP1})")
                if angle_constraints[-1][1] > 0:
                    logger.info(f"g{gid}, c{cid} OP1 outflow angle: constraint violation, should be between {self.angle_OP1[0]} and {self.angle_OP1[1]}")
                logger.debug(f"g{gid}, c{cid} OP2 outflow angle: ({outflow_angle_OP2})")
                if angle_constraints[-1][2] > 0:
                    logger.info(f"g{gid}, c{cid} OP2 outflow angle: constraint violation, should be between {self.angle_OP2[0]} and {self.angle_OP2[1]}")
        print(self.J)
        out["F"] = np.vstack(self.J[-self.doe_size:])
        self._observe(out["F"])
        out["G"] = np.column_stack([geom_constraints, np.vstack(angle_constraints)])
        for g in out["G"]:
            self.G.append(g.tolist())
        self.gen_ctr += 1

    def apply_candidate_constraints(self, profile: np.ndarray, gid: int, cid: int) -> list[float]:
        """
        **Computes** various relative and absolute constraints of a given candidate
        and **returns** their values as a list of floats.

        Note:
            when some constraint is violated, a graph is also generated.
        """
        if not self.constraint:
            return [-1.] * 6
        # relative constraints
        # thmax / c:        +/- 30%
        # Xthmax / c_ax:    +/- 20%
        upper, lower = split_profile(profile)
        c, c_ax = get_chords(profile)
        camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
        th_over_c = thmax / c
        Xth_over_cax = Xthmax / c_ax
        logger.debug(f"th_max = {thmax} m, Xth_max {Xthmax} m")
        logger.debug(f"th_max / c = {th_over_c}, Xth_max / c_ax = {Xth_over_cax}")
        th_cond = abs(th_over_c - self.bsl_th_over_c) / self.bsl_th_over_c - 0.3
        logger.debug(f"th_max / c: {'violated' if th_cond > 0 else 'not violated'} ({th_cond})")
        Xth_cond = abs(Xth_over_cax - self.bsl_Xth_over_cax) / self.bsl_Xth_over_cax - 0.2
        logger.debug(f"Xth_max / c_ax: {'violated' if Xth_cond > 0 else 'not violated'} "
                     f"({Xth_cond})")
        # area / (c * c):   +/- 20%
        area = get_area(profile)
        area_over_c2 = area / c**2
        area_cond = abs(area_over_c2 - self.bsl_area_over_c2) / self.bsl_area_over_c2 - 0.2
        logger.debug(f"area / (c * c): {'violated' if area_cond > 0 else 'not violated'} "
                     f"({area_cond})")
        # X_cg / c_ax:      +/- 20%
        cog = get_cog(profile)
        Xcg_over_cax = cog[0] / c_ax
        cog_cond = abs(Xcg_over_cax - self.bsl_Xcg_over_cax) / self.bsl_Xcg_over_cax - 0.2
        logger.debug(f"X_cg / c_ax: {'violated' if cog_cond > 0 else 'not violated'} ({cog_cond})")
        # absolute constraints
        # leading/trailing edge radii and principal axis condition
        O_le = get_valid_center(
            profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=True
        )
        O_te = get_valid_center(
            profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=False
        )
        le_circle = get_circle(O_le, 0.005 * c) if O_le is not None else np.array([])
        te_circle = get_circle(O_te, 0.005 * c) if O_te is not None else np.array([])
        # leading edge radius: r_le > 0.5% * c
        logger.debug(f"le radius: {'violated' if O_le is None else 'not violated'}")
        le_cond = 1 if O_le is None else -1
        # trailing edge radius: r_te > 0.5% * c
        logger.debug(f"te radius: {'violated' if O_te is None else 'not violated'}")
        te_cond = 1 if O_te is None else -1
        if cog_cond > 0:
            fig_name = os.path.join(self.figdir, f"profile_g{gid}_c{cid}.png")
            plot_profile(profile, cog, fig_name)
        if (th_cond > 0 or Xth_cond > 0 or area_cond > 0 or le_cond > 0 or te_cond > 0):
            fig_name = os.path.join(self.figdir, f"sides_g{gid}_c{cid}.png")
            plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, fig_name)
        return [th_cond, Xth_cond, area_cond, cog_cond, le_cond, te_cond]


class CustomEvolution(WolfCustomEvolution):
    """
    Same custom class as the one defined in cascade_adap.
    """

