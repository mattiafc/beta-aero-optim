import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas 
from typing import Any
import scipy.interpolate as si

# Quadrilateral elements
def create_elm(nb: int, line: int) -> list[list[int]]:
    quad = []
    for i in range(nb):
        bg = i          # bas-gauche
        bd = bg + 1     # bas-droite
        hd = bd + line  # haut-droite
        hg = bg + line  # haut-gauche

        if bd % line != 0:
            quad.append([bg, bd, hd, hg])  # liste des coins des elements
    return quad


# Plotting function for mesh
def plot_elm(nod_x: np.ndarray, nod_y: np.ndarray, qd: list, edgecolor: str):
    for cell in qd:
        vertices_x = [nod_x[cell[i]] for i in range(len(cell))]
        vertices_y = [nod_y[cell[i]] for i in range(len(cell))]
        plt.fill(vertices_x, vertices_y, edgecolor=edgecolor, linewidth=0.3, fill=False)


def read_next_line_in_file(fname: str, pattern: str) -> str:
    """
    Returns the next line of fname containing pattern.
    """
    filedata = open(fname, 'r').readlines()
    index = [idx for idx, s in enumerate(filedata) if pattern in s][0] + 1
    if index:
        return filedata[index].strip()
    else:
        raise Exception(f"{pattern} not found in {fname}")
        
def get_block_info(dat_dir: str) -> dict:
    """
    Returns a dictionnary containing relevant information on the blocks
    used by MUSICAA: number of blocks, block size, snapshots.
    """
    block_info: dict[str, Any] = {}

    # get number of blocks
    filename = os.path.join(dat_dir, "param_blocks.ini")
    nbl_ = read_next_line_in_file(filename, "Number of Blocks")
    nbl = int(re.findall(r"\d+", nbl_)[0])
    block_info["nbl"] = nbl

    # iterate for each block
    total_nb_points = 0
    total_nb_lines = 0
    total_nb_planes = 0
    with open(filename, "r") as f:
        filedata = f.readlines()
    for bl in range(nbl):
        bl += 1
        pattern = f"! Block #{bl}"
        block_info[f"block_{bl}"] = {}
        for i, line in enumerate(filedata):
            if pattern in line:
                block_info[f"block_{bl}"]["nx"] = int(re.findall(r"\d+", filedata[i + 3])[0])
                block_info[f"block_{bl}"]["ny"] = int(re.findall(r"\d+", filedata[i + 4])[0])
                block_info[f"block_{bl}"]["nz"] = int(re.findall(r"\d+", filedata[i + 5])[0])
                # get eventual sensors if any
                nb_snaps = int(re.findall(r"\d+", filedata[i + 16])[0])
                block_info[f"block_{bl}"]["nb_snaps"] = nb_snaps
                block_info[f"block_{bl}"]["nb_points"] = 0
                block_info[f"block_{bl}"]["nb_lines"] = 0
                block_info[f"block_{bl}"]["nb_planes"] = 0
                point_nb = 0
                line_nb = 0
                plane_nb = 0
                if nb_snaps > 0:
                    for snap in range(nb_snaps):
                        snap += 1
                        info_snap = [
                            int(dim) for dim in re.findall(r"\d+", filedata[i + 20 + snap])]
                        nx1, nx2 = info_snap[0], info_snap[1]
                        ny1, ny2 = info_snap[2], info_snap[3]
                        nz1, nz2 = info_snap[4], info_snap[5]
                        freq = info_snap[6]
                        nvars = info_snap[7]
                        # line
                        if (
                            (nx1 == nx2 and ny1 == ny2 and nz1 != nz2)
                            or (nx1 == nx2 and nz1 == nz2 and ny1 != ny2)
                            or (ny1 == ny2 and nz1 == nz2 and nx1 != nx2)
                        ):
                            snap_type = "line"
                            param_freq = 1
                            line_nb += 1
                            total_nb_lines += 1
                            snap_id = line_nb
                        # point
                        elif (nx1 == nx2 and ny1 == ny2 and nz1 == nz2):
                            snap_type = "point"
                            param_freq = 0
                            point_nb += 1
                            total_nb_points += 1
                            snap_id = point_nb
                        # plane
                        else:
                            snap_type = "plane"
                            snap = plane_nb + 1
                            param_freq = 2
                            plane_nb += 1
                            total_nb_planes += 1
                            snap_id = plane_nb
                        if freq == 0:
                            # frequency is prescribed in param.ini
                            freq = int(read_next_line_in_file(
                                os.path.join(dat_dir, "param.ini"),
                                "Snapshot frequencies").split()[param_freq])
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"] = {}
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nx1"] = nx1
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nx2"] = nx2
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["ny1"] = ny1
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["ny2"] = ny2
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nz1"] = nz1
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nz2"] = nz2
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["freq"] = freq
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nvars"] = nvars
                        var_list = filedata[i + 20 + snap].split()[-nvars:]
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["var_list"] = var_list
                        position = point_nb + line_nb + plane_nb
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["position"] = position
                    block_info[f"block_{bl}"]["nb_points"] = point_nb
                    block_info[f"block_{bl}"]["nb_lines"] = line_nb
                    block_info[f"block_{bl}"]["nb_planes"] = plane_nb
    block_info["total_nb_points"] = total_nb_points
    block_info["total_nb_lines"] = total_nb_lines
    block_info["total_nb_planes"] = total_nb_planes
    return block_info

def read_grid(
        dir_data: str, file_input: str
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    print("Reading grid...")
    bl = file_input.split('bl')[-1][0]
    dict_info = read_info(dir_data)
    nx = dict_info[f'nx_bl{bl}']
    ny = dict_info[f'ny_bl{bl}']
    nz = dict_info[f'nz_bl{bl}']
    ngh = dict_info['ngh']
    ngh = 5
    nx_ngh = nx + 2 * ngh
    ny_ngh = ny + 2 * ngh

    f = open(file_input, 'r')
    x = np.fromfile(f, dtype=('<f8'), count=nx_ngh * ny_ngh).reshape((nx_ngh, ny_ngh, 1), order='F')
    y = np.fromfile(f, dtype=('<f8'), count=nx_ngh * ny_ngh).reshape((nx_ngh, ny_ngh, 1), order='F')
    z = np.zeros(1)
    x = x[ngh:-ngh, ngh:-ngh, 0]
    y = y[ngh:-ngh, ngh:-ngh, 0]
    return nx, ny, nz, x, y, z


def read_info(dir_data: str) -> dict:
    dict_info: dict[Any, Any] = {}
    lines = open(os.path.join(dir_data, 'info.ini'), 'r').readlines()
    dict_info["nbloc"] = int(lines[0].split()[4])
    dict_info["is_curv"] = lines[0].split()[5]
    # block info
    for ind in range(dict_info["nbloc"]):
        dict_info["nx_bl" + str(ind + 1)] = int(lines[1 + ind].split()[5])
        dict_info["ny_bl" + str(ind + 1)] = int(lines[1 + ind].split()[6])
        dict_info["nz_bl" + str(ind + 1)] = int(lines[1 + ind].split()[7])
    # sim info
    pattern = r"([A-Za-z0-9 ]+)\s*=\s*(.*)"
    for line in lines[ind + 2:]:
        match = re.match(pattern, line.strip())
        if match:
            keywords_list = match.group(1).strip().split()
            values = match.group(2).split()
            for val_idx in range(len(values)):
                dict_info[keywords_list[val_idx]] = float(values[val_idx])
    return dict_info
    
# Block to be plotted
def plot_grid(
        dir_data: str, is_curv: bool,
        bl_list: list, every: int = 5,
        ngh: int = 5, figsize: tuple[float, float] = (12, 10),
        xlabel: str = "$x$ [mm]", ylabel: str = "$y$ [mm]",
):
    """
    Plots mesh with 3 subplots:
    - Top: full view with zoom boxes highlighted
    - Bottom left: zoom on box1
    - Bottom right: zoom on box2
    """
    # General information
    dict_info = read_info(dir_data)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(gs[0, :])  # Top subplot (full width)
    #ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    #ax3 = fig.add_subplot(gs[1, 1])  # Bottom right
    
    axes = [ax1]
    
    # Plot on all three subplots
    for ax_idx, ax in enumerate(axes):
        plt.sca(ax)  # Set current axis
        
        for bl_num in bl_list:
            bl_file = os.path.join(dir_data, f'grid_bl{bl_num}_ngh{ngh}.bin')
            nx, ny, nz, x, y, z = read_grid(dir_data, bl_file)
            x, y, z = x[::every], y[::every], z
            node_x, node_y = x.flat[::every], y.flat[::every]

            nb_cells = (nx // every - 1) * ny // every
            if is_curv:
                quad = create_elm(nb_cells, ny // every)
                if bl_num == 6:
                    plot_elm(node_x, node_y - 0.621, quad, 'tab:blue')
                else:
                    plot_elm(node_x, node_y, quad, 'tab:blue')
            else:
                ax.vlines(x[0], ymin=y[0, 0], ymax=y[-1, 0], colors='tab:blue', linewidth=1)
                ax.hlines(y[:, 0], xmin=x[0, 0], xmax=x[0, -1], colors='tab:blue', linewidth=1)
            
            # Add block number label in green at the center of the block
            # x_center = (x.min() + x.max()) / 2
            # y_center = (y.min() + y.max()) / 2
            # ax.text(x_center, y_center, str(bl_num), color='green', fontsize=10, 
            #        ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='datalim')

    plt.savefig("plotMesh.png")
    plt.show()
    
def main():
    
    input_dir = "output_mesh/MESH/musicaa_L030_g0_c0"
    output_key = "ADP"
    
    print("Reading simulation information...")
    dict_info = read_info(input_dir)
    figsize = (5.2, 3.64)
    block_info = get_block_info(input_dir)
    
    # For each block, the grid coordinates (x, y, z) and plane data are extracted 
    # and saved in data[bl]
    # Notes: ngh is the number of ghost cells and num_blocks is the total number 
    # of blocks. The blocks are numbered from 1 to num_blocks.
    
    print("Reading grid data...")
    data = {}
    ngh = int(dict_info["ngh"])
    n_block = dict_info["nbloc"]
    
    for bl in range(1, n_block + 1):
        data[bl] = {}
        # read grid
        bl_file = os.path.join(input_dir, f'grid_bl{bl}_ngh{ngh}.bin')
        nx, ny, nz, x, y, z = read_grid(input_dir, bl_file)
        # scale grid
        data[bl]["x"], data[bl]["y"], data[bl]["z"] = x, y, z
    
    # Plot mesh
    print("Plotting mesh...")
    plot_grid(input_dir, True, [3,4,7,6], every=5, figsize=figsize)

if __name__ == "__main__":
    main()