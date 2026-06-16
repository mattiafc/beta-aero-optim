import argparse
import logging
import subprocess
import functools
import numpy as np
import os
import sys
import re
import pandas as pd
import json
from typing import Callable
import time

from custom_cascade_musicaa import (get_time_info, get_niter_ftt, compute_QoIs,
                                   get_mp1_data, save_history, get_history)
from aero_optim.utils import (cp_filelist, rm_filelist, read_next_line_in_file,
                              custom_input, submit_popen_process, wait_for_it)

EPSILON: float = 1e-6
FAILURE: int = 1
SUCCESS: int = 0
MUSICAA: str = "mpiexec -n @nproc /home/nparini/beta-aero-optim/examples/nparini/musicaa"

print = functools.partial(print, flush=True)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def get_nproc(sim_outdir: str) -> int:
    """
    **Computes** the number of proc required by the simulation from param_blocks.ini.
    """
    f = open(os.path.join(sim_outdir, "param_blocks.ini"), "r").read().splitlines()
    nproc = 0
    for ll, line in enumerate(f):
        if "Nb procs" in line:
            nproc += np.prod(np.array([int(re.findall(r'\d+', f[ll + j])[1]) for j in range(1, 4)]))
    return nproc

def monitor_sim_progress(proc: subprocess.Popen):
    """
    **Monitors** a simulation.
    """

    istamp = 0
    while True:
        returncode = proc.poll()
        # computation still running
        if returncode is None:
            if istamp == 0:
                print("INFO -- simulation running")
                istamp = 1
        # computation has crashed
        elif returncode > 0:
            raise Exception(f"ERROR -- RANS simulation crashed")
        # computation has completed
        elif returncode == 0:
            print("INFO -- simulation completed")
            break
    

def pre_process_stats(config: dict, sim_outdir: str, computation_type: str):
    """
    **Pre-processes** computation for statistics.
    """

    # get current iteration from time.ini
    time_info = get_time_info(sim_outdir)
    ndeb_stats = time_info["niter_total"]

    # modify param.ini file
    args = {}
    args.update({"from_field": "2"})
    args.update({"Iteration number to start statistics": f"{ndeb_stats + 1}"})
    param_ini = os.path.join(sim_outdir, "param.ini")
    if computation_type == "steady":
        niter_stats = 1
    elif computation_type == "unsteady":
        print("HERE")
        niter_stats = 999999
        # add frequency for QoI convergence check: will ask MUSICAA to output stats files
        niter_ftt = get_niter_ftt(sim_outdir, config["gmsh"]["chord_length"])
        freqs = read_next_line_in_file(param_ini,
                                       "Output frequencies: screen / stats / fields").split()
        args.update({"Output frequencies: screen / stats / fields":
                     f"{freqs[0]} {freqs[1]} {niter_ftt}"})
    args.update({"Max number of temporal iterations": f"{niter_stats} 3000.0"})
    custom_input(param_ini, args)

def pre_process_post(config: dict, sim_outdir: str):
    """
    **Pre-processes** computation for MUSICAA postprocessing.
    """
    args = {}
    param_ini = os.path.join(sim_outdir, "param.ini")
    #
    args.update({"from_interp": "4"})
    #
    custom_input(param_ini, args)

def pre_process_PVAR(config: dict, sim_outdir: str, computation_iter: int):
    """
    **Pre-processes** RANS_PVAR computation, changed backpressure BC value at each iteration.
    """
    args = {}
    param_ini = os.path.join(sim_outdir, "param.ini")
    stop_file = os.path.join(sim_outdir, f"../stop{sim_outdir}")
    # Recover restarting related parameters from config file
    tol = config["simulator"]["restart_criteria"]["mach_number_tol"]
    n_iter_01 = config["simulator"]["restart_criteria"]["n_iter_first"]
    n_iter_02 = config["simulator"]["restart_criteria"]["n_iter_later"]
    n_iter_i = config["simulator"]["restart_criteria"]["n_iter_later"]
    if computation_iter == 0:
        # --- RUN CASE 01: Simulation start from scratch ---
        args.update({
            'Max number of temporal iterations': f"{n_iter_01} 3000.0",
            "from_interp": "1"})
        # Calculate initial P_ex from Reference Total Pressure
        M1 = config["simulator"]["restart_criteria"]["Mtarget"]
        P1_tot = config["simulator"]["restart_criteria"]["Pref"]
        pi = config["simulator"]["restart_criteria"]["pi"]
        # Hard-coded constants (TO CHANGE)
        gam, gam1 = 1.4, 0.4
        cc = (1 + (gam1 / 2) * (M1 ** 2)) ** (gam / gam1)
        P1 = P1_tot / cc
        P2 = P1 * pi
        # HARD CODED P2 TEST - TO CHANGE
        P2 = 79718
        args.update({'back-pressure': P2})
    elif computation_iter == 1:
        # --- RUN CASE 02: Simulation start from 01 case ---
        args.update({
            'Max number of temporal iterations': f"{n_iter_02} 3000.0",
            'from_interp': 2
        })
        # Hard-coded constants (TO CHANGE)
        M_target = config["simulator"]["restart_criteria"]["Mtarget"]
        alpha = 10000
        # Load and log historical data
        mp1_data = get_mp1_data(sim_outdir)
        M1_01 = mp1_data["mach"] 
        P2_01 = mp1_data["pressure_back"] 
        n_partial = mp1_data["n_partial"] 
        err = np.abs(M1_01 - M_target)
        # Save historical data of previous iteration
        history = {
            "n_partial": n_partial,
            "mach": M1_01,
            "pressure": P2_01,
            "error": err
        }
        save_history(sim_outdir, history)
        # Proportional correction for the next step
        P2_02 = P2_01 - alpha * (M_target - M1_01)
        args.update({'back-pressure': P2_02})
    else:
        # --- RUN CASE i: Standard Iterative Loop ---
        args.update({
            'Max number of temporal iterations': f"{n_iter_i} 3000",
            'from_interp': 2
        })
        # Load current step data and history
        M_target = config["simulator"]["restart_criteria"]["Mtarget"]
        mp1_data = get_mp1_data(sim_outdir)
        M_latest = mp1_data["mach"] 
        P_latest = mp1_data["pressure_back"] 
        n_partial = mp1_data["n_partial"]
        hist_data = get_history(sim_outdir)
        M_hist = hist_data["mach"] 
        P_hist = hist_data["pressure"] 
        err = np.abs(M_latest - M_target)
        # Save historical data of previous iteration
        history = {
            "n_partial": n_partial,
            "mach": M_latest,
            "pressure": P_latest,
            "error": err
        }
        save_history(sim_outdir, history)
        
        # Check convergence tolerance
        if err < tol:
            try:
                open(stop_file, "x").close()  
            except FileExistsError:
                pass 
            return
            
        # Update for the new backpressure
        M_m1, M_m2 = M_latest, M_hist[-1]
        P_m1, P_m2 = P_latest, P_hist[-1]
        P2_new = P_m1 + (P_m1 - P_m2) * ((M_target - M_m1) / (M_m1 - M_m2))
        args.update({'back-pressure': P2_new})
    # modify param.ini file
    custom_input(param_ini, args)

def execute_steady(config: dict, sim_outdir: str):
    """
    **Executes** a Reynolds-Averaged Navier-Stokes simulation with MUSICAA. RANS_PVAR loop implemented,
    simulation is automatically restarted until the inlet mach number as reached a desired target 
    value (or the number of maximum restart is reached).
    """
    # start RANS_PVAR loop
    current_restart = 0
    config.update({"is_stats": False})
    config.update({"is_post": False})
    while True:
        # prepare the input files 
        pre_process_PVAR(config, sim_outdir, current_restart)
        exec_cmd = MUSICAA.replace("@nproc", str(get_nproc(sim_outdir)))
        print(f"INFO -- submit popen steady: {exec_cmd} - ITER {current_restart}")
        _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
        monitor_sim_progress(proc)
        current_restart += 1
        stop_file = os.path.join(sim_outdir, f"../stop{sim_outdir}")
        # exit the loop when the number of max iter is reached or convergence is reached
        if (current_restart >= config["simulator"]["restart_criteria"]["iter_max"]) or os.path.exists(stop_file):
            # Clean up the stop file if it exists so it doesn't affect subsequent runs
            if os.path.exists(stop_file):
                os.remove(stop_file)
            break

    # gather statistics
    config.update({"is_stats": True})
    pre_process_stats(config, sim_outdir, "steady")
    nproc = get_nproc(sim_outdir)
    exec_cmd = MUSICAA.replace("@nproc", str(nproc))
    print(f"INFO -- submit popen steady: {exec_cmd} - STATS")
    _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
    monitor_sim_progress(proc)

    # compute postprocess 
    config.update({"is_post": True})
    pre_process_post(config, sim_outdir)
    nproc = 9
    exec_cmd = MUSICAA.replace("@nproc", str(nproc))
    print(f"INFO -- submit popen steady: {exec_cmd} - POST-PROCESSING")
    _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
    monitor_sim_progress(proc)

    # compute QoIs and save to file
    new_QoIs_df = compute_QoIs(config, sim_outdir)
    filename = os.path.join(sim_outdir, "QoI_convergence.csv")
    if not os.path.isfile(filename):
        # first time computing the QoIs
        new_QoIs_df.to_csv(filename, index=False)
        return False
    QoIs_df = pd.concat([pd.read_csv(filename), new_QoIs_df], axis=0)
    QoIs_df.to_csv(filename, index=False)
    QoIs = QoIs_df[config["optim"]["QoI"]].to_numpy()


def main() -> int:
    """
    This program runs a MUSICAA RANS simulation at ADP.
    In multisimulation mode, OP1 (+0.5°) and OP2 (-0.5°) simulations are also executed.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ms", type=int, help="number of parallel simulations", default=-1)
    parser.add_argument("-adp", action="store_true", help="simulate ADP")
    parser.add_argument("-op1", action="store_true", help="simulate OP1")
    parser.add_argument("-op2", action="store_true", help="simulate OP2")

    args_parse = parser.parse_args()
    t0 = time.time()
    print(f"simulations performed with: {args_parse}\n")

    # read config file
    with open("sim_config.json") as jfile:
        config = json.load(jfile)
    # set computation type 
    # ONLY steady working 
    computation_type: str = read_next_line_in_file("param.ini", "DES without subgrid scale modelling ('DES')")
    computation_type = "unsteady" if computation_type == "N" else "steady"
    execute_computation: Callable = globals()[f"execute_{computation_type}"]

    if args_parse.ms > 0:
        # tracking variables
        l_proc = []
        # process cmd line
        exec_args = sys.argv
        ms_idx = exec_args.index("-ms")
        del exec_args[ms_idx + 1]
        del exec_args[ms_idx]
        # adp
        exec_cmd = [sys.executable] + exec_args + ["-adp"]
        l_proc.append(submit_popen_process("ADP", exec_cmd))
        # op1
        exec_cmd = [sys.executable] + exec_args + ["-op1"]
        if len(l_proc) < args_parse.ms:
            l_proc.append(submit_popen_process("OP1", exec_cmd))
        else:
            wait_for_it(l_proc, args_parse.ms)
            l_proc.append(submit_popen_process("OP1", exec_cmd))
        # op2
        exec_cmd = [sys.executable] + exec_args + ["-op2"]
        if len(l_proc) < args_parse.ms:
            l_proc.append(submit_popen_process("OP2", exec_cmd))
        else:
            wait_for_it(l_proc, args_parse.ms)
            l_proc.append(submit_popen_process("OP2", exec_cmd))
        # wait for all processes to finish
        wait_for_it(l_proc, 1)

    if args_parse.adp:
        print("** ADP SIMULATION **")
        print("** -------------- **")
        sim_dir = "ADP"
        os.makedirs(sim_dir, exist_ok=True)
        args = {}
        # add simulation files
        cp_filelist(config["simulator"]["cp_list"], [sim_dir] * len(config["simulator"]["cp_list"]))
        # specify path for mesh files
        old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
        dir_grid = "'" + os.path.join("../", old_dir_grid) + "'"
        args.update({"Directory for grid files": dir_grid})
        param_ini = os.path.join(sim_dir, "param.ini")
        custom_input(param_ini, args)
        # execute computation
        config_ADP = config.copy()
        execute_computation(config_ADP, sim_dir)

    if args_parse.op1:
        print("** OP1 SIMULATION (+0.5 deg.) **")
        print("** ------------------------ **")
        sim_dir = "OP1"
        os.makedirs(sim_dir, exist_ok=True)
        # add simulation files
        cp_filelist(config["simulator"]["cp_list"], [sim_dir] * len(config["simulator"]["cp_list"]))
        args = {}
        # specify path for mesh files
        old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
        dir_grid = "'" + os.path.join("../", old_dir_grid) + "'"
        args.update({"Directory for grid files": dir_grid})
        # change flow angle
        args.update({"Flow angles": "59. 0."})
        param_ini = os.path.join(sim_dir, "param.ini")
        custom_input(param_ini, args)
        # execute computation
        config_OP1 = config.copy()
        execute_computation(config_OP1, sim_dir)

    if args_parse.op2:
        print("** OP2 SIMULATION (-0.5 deg.) **")
        print("** ------------------------ **")
        sim_dir = "OP2"
        os.makedirs(sim_dir, exist_ok=True)
        # add simulation files
        cp_filelist(config["simulator"]["cp_list"], [sim_dir] * len(config["simulator"]["cp_list"]))
        args = {}
        # specify path for mesh files
        old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
        dir_grid = "'" + os.path.join("../", old_dir_grid) + "'"
        args.update({"Directory for grid files": dir_grid})
        # change flow angle
        args.update({"Flow angles": "58. 0."})
        param_ini = os.path.join(sim_dir, "param.ini")
        custom_input(param_ini, args)
        # execute computation
        config_OP2 = config.copy()
        execute_computation(config_OP2, sim_dir)

    print(f"INFO -- simulations finished successfully in {time.time() - t0} seconds.")
    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
