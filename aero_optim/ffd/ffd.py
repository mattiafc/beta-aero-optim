import logging
import math
from Cython import profile
import numpy as np
import os
import shutil
import copy

from abc import ABC, abstractmethod
from scipy.stats import qmc
from typing import Any,Tuple,List,Union
from collections import OrderedDict
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import pandas as pd

from parablade.blade_3D import Blade3D
from parablade.common.config import ReadUserInput

# from sklearn.calibration import delayed

from aero_optim.geom import get_cog
from aero_optim.utils import from_dat, check_dir

logger = logging.getLogger(__name__)


class Deform(ABC):
    """
    This class implements an abstract Deform class.
    """
    def __init__(self, dat_file: str, ncontrol: int, header: int = 2, scale: float = 1, **kwargs):
        """
        Instantiates the abstract Deform object.

        **Input**

        - dat_file (str): path to input_geometry.dat.
        - ncontrol (int): the number of control points.
        - header (int): the number of header lines in dat_file.
        - scale (float): the geometry scaling factor

        **Inner**

        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.
        """
        self.dat_file: str = dat_file
        self.pts: np.ndarray = np.array(from_dat(self.dat_file, header, scale))
        self.ncontrol = ncontrol

    def write_ffd(
            self,
            profile: np.ndarray,
            Delta: np.ndarray,
            outdir: str,
            gid: int = 0, cid: int = 0
    ) -> str:
        """
        **Writes** the deformed geometry to file and **returns** /path/to/outdir/outfile.

        - profile (np.ndarray): the deformed geometry coordinates to be written to outfile.
        - Delta (np.ndarray): the deformation vector.
        - outdir (str): the output directory (it is to be combined with outfile).
        """
        outfile = f"{self.dat_file.split('/')[-1][:-4]}_g{gid}_c{cid}.dat"
        check_dir(outdir)
        logger.info(f"write profile g{gid} c{cid} as {outfile} to {outdir}")
        np.savetxt(os.path.join(outdir, outfile), profile,
                   header=f"Deformed profile {outfile}\nDelta={[d for d in Delta]}")
        return os.path.join(outdir, outfile)

    @abstractmethod
    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        """
        Returns a deformed profile.
        """

class DLR(ABC):
    """
    This class implements an abstract Deform class.
    """
    def __init__(self, dat_file: str, home_dir: str, bladegen_path: str, param_bounds: dict, header: int = 2, scale: float = 1, **kwargs):
        """
        Instantiates the abstract Deform object.

        **Input**

        - dat_file (str): path to input_geometry.dat.
        - ncontrol (int): the number of control points.
        - header (int): the number of header lines in dat_file.
        - scale (float): the geometry scaling factor

        **Inner**

        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.
        """

        # self.home_dir : str = home_dir
        # self.bladegen_path: str = bladegen_path
        # self.param_bounds: dict = param_bounds

        # self.header, self.entries = self.read_progen_input()
        # self.run_blade_generator(self.home_dir)
        # self.pts = self.extract_data_from_tec(os.path.join(self.home_dir, 'Output', 'Profile_mergedGestaf.tec'))

        # print(self.pts)
        # input()

        self.home_dir = home_dir
        self.bladegen_path: str = bladegen_path
        self.param_bounds: dict = param_bounds
        self.scale = scale

        self.header, self.entries = self.read_progen_input()
        temp = np.array(from_dat(dat_file, header))
        temp = np.vstack([temp, temp[0]])
        if temp.shape[1] == 3:
            self.pts = temp[:,:-1]
        elif temp.shape[1] == 2:
            self.pts = temp
        else:
            raise ValueError(f"Unexpected shape {temp.shape} for points array.")

    def read_progen_input(self) -> None:
        """
        Legge il file `progen.input`, salva le prime 3 righe in `header`
        e il resto in un OrderedDict `entries`.
        - Commenti (linee che iniziano con '#' o '!') e linee vuote sono conservate
        come chiavi speciali '__comment_N' o '__blank_N' con valore = riga originale.
        - Linee con '=' vengono parse come key = value (split sulla prima '=').
        - Altre linee vengono parse come "key [value...]" (primo token = key, resto = value).
        - Se una chiave appare più volte, il valore viene salvato come lista di (sep, value) tuples per preservare il separatore.
        Ritorna (header_lines, entries).
        """
        with open(os.path.join(self.home_dir, 'progen.input'), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header = [lines[i].rstrip('\n') for i in range(min(3, len(lines)))]
        rest = lines[3:] if len(lines) > 3 else []

        entries: OrderedDict[str, Union[Tuple[Union[str, None], str], List[Tuple[Union[str, None], str]]]] = OrderedDict()
        comment_idx = 0
        blank_idx = 0

        for raw in rest:
            line = raw.rstrip('\n')
            stripped = line.lstrip()
            if stripped == '':
                blank_idx += 1
                entries[f'__blank_{blank_idx}'] = ''  # rappresenta una linea vuota
                continue
            if stripped.startswith('#') or stripped.startswith('!'):
                comment_idx += 1
                entries[f'__comment_{comment_idx}'] = line
                continue

            # detect separator to preserve formatting: '=' or whitespace
            if '=' in line:
                key, val = line.split('=', 1)
                sep = '='
                key = key.strip()
                val = val.strip()
            else:
                parts = line.split(None, 1)
                sep = None
                key = parts[0].strip()
                val = parts[1].strip() if len(parts) > 1 else ''

            item = (sep, val)
            if key in entries:
                existing = entries[key]
                if isinstance(existing, list):
                    existing.append(item)
                else:
                    entries[key] = [existing, item]
            else:
                entries[key] = item

        return header, entries

    def write_progen_input(self, out_dir: str, input_dict: dict, no_rewriting: bool = False) -> None:
        """
        Scrive su file con lo stesso formato di progen.input:
        - Scrive le prime 3 righe (header) così come sono.
        - Poi scrive le entries nell'ordine dato:
        * '__blank_N' -> riga vuota
        * '__comment_N' -> riga commento così com'è
        * key -> 'key = value' o 'key value' a seconda del separatore originale
        out_dir must be a directory path; the function will create it if needed and write 'progen.input' inside.
        """

        if no_rewriting == True and os.path.exists(os.path.join(out_dir, 'progen.input')):
                # print("All values are within the bounds. No need to rewrite progen.input")
                return

            # _, entries = self.read_progen_input()

            # # Verifica che i valori comuni siano nei bounds
            # same_bounds = True
            # for key in self.param_bounds.keys():
            #     val = float(entries[key][1])  # entries[key] è una tupla (sep, value)
            #     min_bound, max_bound = self.param_bounds[key]
            #     if not (min_bound <= val <= max_bound):
            #         same_bounds = False

            # if same_bounds == True:
            #     # input()

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, 'progen.input')

        with open(out_file, 'w', encoding='utf-8') as f:
            # header
            for i in range(3):
                line = self.header[i] if i < len(self.header) else ''
                f.write(line.rstrip('\n') + '\n')

            # body
            for key, val in self.entries.items():
                if key.startswith('__blank_'):
                    f.write('\n')
                elif key.startswith('__comment_'):
                    # val contiene la riga così com'era
                    f.write(val.rstrip('\n') + '\n')
                else:
                    # entries store either (sep, value) or list of (sep, value)
                    items = val if isinstance(val, list) else [val]

                    # If input_dict provided and key present in input_dict, replace values
                    if input_dict is not None and key in input_dict:
                        ov = input_dict[key]
                        # choose separator preference from existing items (prefer first item's sep)
                        first_sep = items[0][0] if items else None
                        if isinstance(ov, list):
                            # build items list from input_dict list
                            items = [(first_sep, str(x)) for x in ov]
                        else:
                            items = [(first_sep, str(ov))]

                    for sep, v in items:
                        if sep == '=':
                            f.write(f"{key} = {v}\n")
                        else:
                            # write with single space separator to mimic original whitespace format
                            if v == '':
                                f.write(f"{key}\n")
                            else:
                                f.write(f"{key} {v}\n")

    def run_blade_generator(self, variant_dir: str) -> None:
        """Function for running BladeGenerator"""
        if os.path.exists(os.path.join(variant_dir, 'Output', 'Profile_mergedGestaf.tec')):
            return
        try:
            # Change to variant directory and run BladeGenerator
            current_dir = os.getcwd()
            os.chdir(variant_dir)

            # Run BladeGenerator using os.system
            returncode = os.system(self.bladegen_path)

            # Change back to original directory
            os.chdir(current_dir)

            return (variant_dir, returncode, "", "")
        except Exception as e:
            if 'current_dir' in locals():
                os.chdir(current_dir)
            logger.info(f"ERRORE {e}")

    def extract_data_from_tec(self, file_path, zone_name="NURB_Profil_gestaf0") -> np.ndarray:
        x_values = []
        y_values = []
        in_zone = False

        with open(file_path, 'r') as file:
            for line in file:
                if f'ZONE T="{zone_name}"' in line:
                    in_zone = True  # Start reading the data for the desired zone
                elif in_zone and line.strip().startswith("ZONE"):
                    in_zone = False  # Stop reading after the zone ends
                elif in_zone:
                    # Extract m and theta from the current line (first two columns)
                    try:
                        values = line.split()
                        x_values.append(float(values[0]))
                        y_values.append(float(values[1]))
                    except ValueError:
                        continue  # Skip any lines that don't contain valid float values

        # Remove duplicate last point (first and last points are identical in tecplot files)
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        if np.allclose(x_array[0], x_array[-1]) and np.allclose(y_array[0], y_array[-1]):
            x_array = x_array[:-1]
            y_array = y_array[:-1]

        return np.column_stack((x_array, y_array))

    def generate_dict(self, Delta: np.ndarray) -> dict:

        params_dict = {}
        cont = 0
        for k in self.param_bounds.keys():
            params_dict[k] = Delta[cont]
            cont += 1

        return params_dict

    def interp_surface(self, surface, nodes, kind='linear'):
        """
        Interpola una superficie (dorso o ventre) sui nodi specificati.

        Args:
            surface: array 2D con coordinate (x, y) della superficie da interpolare
            nodes: array 1D con le coordinate x dei nodi target per l'interpolazione
            kind: tipo di interpolazione ('linear', 'cubic', etc.)

        Returns:
            array 2D con coordinate (x, y) interpolate sui nodi specificati
        """
        # Ensure array shape
        surface = np.asarray(surface)

        x = surface[:, 0]
        y = surface[:, 1]

        # Sort by x to ensure monotonic x for interpolation
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        f = interp1d(x_sorted, y_sorted, kind=kind, bounds_error=False, fill_value='extrapolate')
        return np.column_stack((nodes, f(nodes)))

    def split_airfoil(self, prof):
            """Separa un profilo in dorso e ventre identificando i punti di cambio direzione x."""
            x_coords = prof[:, 0]
            x_diff = np.diff(x_coords)
            signs = np.sign(x_diff)
            sign_changes = np.diff(signs)
            change_indices = np.where(sign_changes != 0)[0]

            # Clean up change_indices per gestire cambi multipli vicini
            if len(change_indices) > 2:
                cleaned_indices = []
                used = np.zeros(len(change_indices), dtype=bool)

                for i, idx in enumerate(change_indices):
                    if used[i]:
                        continue
                    group = [j for j, other_idx in enumerate(change_indices)
                            if abs(other_idx - idx) <= 3 and not used[j]]
                    for j in group:
                        used[j] = True
                    group_indices = [change_indices[j] for j in group]
                    group_x_diffs = [abs(x_diff[gi]) for gi in group_indices]
                    best_in_group = group_indices[np.argmin(group_x_diffs)]
                    cleaned_indices.append(best_in_group)
                change_indices = np.array(cleaned_indices)

            # Assicurati di avere esattamente 2 punti di cambio (LE e TE)
            if len(change_indices) != 2:
                le_idx = np.argmin(x_coords)
                te_idx = np.argmax(x_coords)
                change_indices = np.array([le_idx-1, te_idx-1]) if le_idx > 0 and te_idx > 0 else np.array([0, len(x_diff)-2])
                change_indices = np.clip(change_indices, 0, len(x_diff)-1)

            assert len(change_indices) == 2, "Impossibile identificare LE e TE del profilo"

            split_indices = np.sort(change_indices + 1)

            # Separa in due zone
            zone_1 = prof[split_indices[0]+1:split_indices[1]+1, :].copy()
            zone_1 = zone_1[np.argsort(zone_1[:, 0], kind='stable')][:-1]

            zone_2 = np.vstack((prof[:split_indices[0]+1, :], prof[split_indices[1]+1:, :])).copy()
            zone_2 = zone_2[np.argsort(-zone_2[:, 0], kind='stable')][:-1]

            # Identifica dorso (upper) e ventre (lower) dalla media y
            if np.mean(zone_2[:,1]) > np.mean(zone_1[:,1]):
                lower, upper = zone_1, zone_2
            else:
                upper, lower = zone_1, zone_2

            return upper, lower, split_indices

    def interpolate_profile(self, profile, original_profile, kind='linear'):
        """
        Interpola un profilo generico usando i nodi definiti da un profilo originale.
        Separa entrambi i profili in dorso e ventre, poi interpola ciascuna superficie
        sui nodi corrispondenti del profilo originale.

        Args:
            profile: 2D array (x, y) del profilo da interpolare (nuovo profilo)
            original_profile: 2D array (x, y) del profilo che definisce i nodi di interpolazione
            kind: tipo di interpolazione ('linear', 'cubic', etc.)

        Returns:
            2D array con profilo interpolato che mantiene esattamente le coordinate x del profilo originale
        """

        # CASO SPECIALE: Se profile e original_profile sono lo stesso (auto-interpolazione)
        # restituisci direttamente una copia per evitare errori numerici
        if np.array_equal(profile, original_profile):
            return original_profile.copy()

        # Calcola min e max del profilo originale
        x_min_orig = np.min(original_profile[:, 0])
        x_max_orig = np.max(original_profile[:, 0])

        # Scala il profilo nuovo per matchare il range del profilo originale
        x_min_new = np.min(profile[:, 0])
        x_max_new = np.max(profile[:, 0])

        # Crea una copia del profilo da scalare
        profile_scaled = profile.copy()

        # Scala le coordinate x del nuovo profilo per matchare il range dell'originale
        if x_max_new != x_min_new:  # Evita divisione per zero
            profile_scaled[:, 0] = (profile[:, 0] - x_min_new) / (x_max_new - x_min_new) * (x_max_orig - x_min_orig) + x_min_orig

        # Separa il profilo nuovo (da interpolare) dopo lo scaling
        new_upper, new_lower, _ = self.split_airfoil(profile_scaled)

        # Separa il profilo originale (che fornisce i nodi di interpolazione)
        orig_upper, orig_lower, orig_split_indices = self.split_airfoil(original_profile)

        # OTTIMIZZAZIONE: Crea le funzioni di interpolazione una sola volta invece che ad ogni iterazione
        # Prepara le superfici ordinate per x
        new_upper_sorted = new_upper[np.argsort(new_upper[:, 0])]
        new_lower_sorted = new_lower[np.argsort(new_lower[:, 0])]
        orig_upper_sorted = orig_upper[np.argsort(orig_upper[:, 0])]
        orig_lower_sorted = orig_lower[np.argsort(orig_lower[:, 0])]

        # Crea funzioni di interpolazione (una volta sola)
        f_new_upper = interp1d(new_upper_sorted[:, 0], new_upper_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
        f_new_lower = interp1d(new_lower_sorted[:, 0], new_lower_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
        f_orig_upper = interp1d(orig_upper_sorted[:, 0], orig_upper_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
        f_orig_lower = interp1d(orig_lower_sorted[:, 0], orig_lower_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')

        # Crea un profilo interpolato con le stesse coordinate x del profilo originale
        interpolated = np.zeros_like(original_profile)
        interpolated[:, 0] = original_profile[:, 0]  # Copia esattamente le coordinate x

        # OTTIMIZZAZIONE: Interpola tutti i punti in una volta (vectorizzato)
        x_coords = original_profile[:, 0]
        y_coords_orig = original_profile[:, 1]

        # Calcola y su entrambe le superfici per tutti i punti
        y_upper_orig = f_orig_upper(x_coords)
        y_lower_orig = f_orig_lower(x_coords)
        y_upper_new = f_new_upper(x_coords)
        y_lower_new = f_new_lower(x_coords)

        # Determina per ogni punto se appartiene a dorso o ventre (operazione vectorizzata)
        dist_to_upper = np.abs(y_coords_orig - y_upper_orig)
        dist_to_lower = np.abs(y_coords_orig - y_lower_orig)
        is_upper = dist_to_upper < dist_to_lower

        # Assegna i valori interpolati (operazione vectorizzata)
        interpolated[:, 1] = np.where(is_upper, y_upper_new, y_lower_new)

        # Post-processing: correggi punti vicini al trailing edge che sono stati classificati male
        # OTTIMIZZAZIONE: Calcola distanze in modo vectorizzato
        distances = np.sqrt(np.diff(interpolated[:, 0])**2 + np.diff(interpolated[:, 1])**2)
        mean_distance = np.mean(distances)
        threshold = mean_distance * 0.03

        # Identifica coppie di punti consecutivi troppo vicini
        close_pairs = np.where(distances < threshold)[0]

        # Se non ci sono coppie problematiche, restituisci subito
        if len(close_pairs) == 0:
            return interpolated

        # Filtra solo le coppie vicine agli split indices
        near_split_mask = np.array([
            (abs(idx - orig_split_indices[0]) < 3 or
            abs(idx - orig_split_indices[1]) < 3 or
            abs(idx + 1 - orig_split_indices[0]) < 3 or
            abs(idx + 1 - orig_split_indices[1]) < 3)
            for idx in close_pairs
        ])
        close_pairs = close_pairs[near_split_mask]

        # Se non ci sono coppie problematiche vicino agli split, restituisci subito
        if len(close_pairs) == 0:
            return interpolated

        # Per le coppie rimanenti, applica la correzione (necessario loop per has_intersection)
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def has_intersection(prof, idx1, idx2):
            """Verifica se modificare i,j crea intersezioni con altri segmenti"""
            segs = [(idx1-1, idx1), (idx1, idx2), (idx2, idx2+1)]
            segs = [(a, b) for a, b in segs if 0 <= a < len(prof) and 0 <= b < len(prof)]
            for a, b in segs:
                p1, p2 = prof[a], prof[b]
                for k in range(len(prof) - 1):
                    if abs(k - a) <= 1 or abs(k - b) <= 1:
                        continue
                    p3, p4 = prof[k], prof[k + 1]
                    if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
                        return True
            return False

        # Precalcola tutti i valori y necessari per le correzioni (vectorizzato)
        x_vals = interpolated[close_pairs, 0]
        x_vals_next = interpolated[close_pairs + 1, 0]

        y_upper_vals = f_new_upper(x_vals)
        y_lower_vals = f_new_lower(x_vals)
        y_upper_vals_next = f_new_upper(x_vals_next)
        y_lower_vals_next = f_new_lower(x_vals_next)

        for pair_idx, idx in enumerate(close_pairs):
            i = idx
            j = idx + 1

            y_i_upper = y_upper_vals[pair_idx]
            y_i_lower = y_lower_vals[pair_idx]
            y_j_upper = y_upper_vals_next[pair_idx]
            y_j_lower = y_lower_vals_next[pair_idx]

            # Testa configurazioni alternative
            configs = [
                (y_i_upper, y_j_lower, abs(y_i_upper - y_j_lower)),
                (y_i_lower, y_j_upper, abs(y_i_lower - y_j_upper))
            ]

            current_dist = abs(interpolated[i, 1] - interpolated[j, 1])
            best_config = None
            best_dist = current_dist

            for yi, yj, dist in configs:
                test_prof = interpolated.copy()
                test_prof[i, 1], test_prof[j, 1] = yi, yj
                if not has_intersection(test_prof, i, j) and dist > best_dist * 1.5:
                    best_config = (yi, yj)
                    best_dist = dist

            if best_config:
                interpolated[i, 1], interpolated[j, 1] = best_config

        return interpolated

    def write_ffd(self, profile: np.ndarray, Delta: np.ndarray, outdir: str, gid: int = 0, cid: int = 0) -> str:
        """
        **Writes** the deformed geometry to file and **returns** /path/to/outdir/outfile.

        - profile (np.ndarray): the deformed geometry coordinates to be written to outfile.
        - Delta (np.ndarray): the deformation vector.
        - outdir (str): the output directory (it is to be combined with outfile).
        """

        outfile = f"{self.dat_file.split('/')[-1][:-4]}_g{gid}_c{cid}.dat"
        check_dir(outdir)
        logger.info(f"write profile g{gid} c{cid} as {outfile} to {outdir}")
        np.savetxt(os.path.join(outdir, outfile), profile[:-1,:],
                   header=f"Deformed profile {outfile}\nParams={[d for d in Delta]}")
        return os.path.join(outdir, outfile)
    
class ParaBlade(ABC):
    """
    This class implements an abstract ParaBlade class.
    """
    def __init__(self, dat_file: str, baseline_dir: str, param_bounds: dict, header: int = 2, scale: float = 1, **kwargs):
        """
        Instantiates the abstract ParaBlade object.
        **Input**
        - dat_file (str): path to input_geometry.dat.
        - baseline_dir (str): path to folder containing baseline matched parametrization/coordinates obtained with ParaBlade BladeMatch.
        - param_bounds (dict): dictionary of the parameters' bounds.
        - header (int): the number of header lines in dat_file.
        - scale (float): the geometry scaling factor
        **Inner**
        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.
        """
        self.baseline_dir = baseline_dir
        self.param_bounds: dict = param_bounds
        self.scale = scale
        self.dat_file: str = dat_file

        self.config_baseline = os.path.join(self.baseline_dir, 'matched_parametrization.cfg')
        self.coordinates_baseline = pd.read_csv(os.path.join(self.baseline_dir, 'matched_coordinates.csv'), sep=',\t')

        temp = np.array(from_dat(dat_file, header))

        if temp.shape[1] == 3:
            self.pts = temp[:,:-1]
        elif temp.shape[1] == 2:
            self.pts = temp[:,:]
        else:
            raise ValueError(f"Unexpected shape {temp.shape} for points array.")

    def generate_dict(self, Delta: np.ndarray) -> dict:
        """
        **Generates** dictionary of parameters associated to the case, **reads** valus from Delta array.
        """
        params_dict = {}
        cont = 0
        for k in self.param_bounds.keys():
            params_dict[k] = Delta[cont]
            cont += 1

        return params_dict
    
    def write_config_input(self, out_dir: str, input_dict: dict, no_rewriting: bool = False) -> None:
        """
        **Manages** modifying of config (.cfg) file for the specific case.
        """
        if no_rewriting == True and os.path.exists(os.path.join(out_dir, 'parablade_input.cfg')):
            return

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, 'parablade_input.cfg')
        self.update_config_file(input_dict, out_file)

    def update_config_file(self, updates, file_output):
        """
        **Reads** a .cfg file, updates specific parameters, and saves it back.
    
        """
        updated_lines = []
        
        # Read the original file and modify the target lines
        with open(self.config_baseline, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                
                # Skip empty lines or comments
                if not stripped_line or stripped_line.startswith('#'):
                    updated_lines.append(line)
                    continue
                
                # Split by the first '=' found
                if '=' in stripped_line:
                    key, value = stripped_line.split('=', 1)
                    key = key.strip()
                    
                    # If this key is in our updates dictionary, modify it
                    if key in updates:
                        new_value = updates[key]
                        line = f"{key}={new_value}\n"
                        
                updated_lines.append(line)
                
        # Write the modified lines back to the file
        with open(file_output, 'w') as file:
            file.writelines(updated_lines)
        
    def extract_data(self, out_dir: str) -> np.ndarray:
        """
        **Extracts** x-y arrays coordinates from the Blade3D (ParaBlade) object generated 
        from the config file of the specific case. **Writes** profile array (n_points_on_profile X 2).
        """
        config_file = os.path.join(out_dir, 'parablade_input.cfg')
        config = ReadUserInput(config_file)
        blade_object = Blade3D(config)
        profile_blade = blade_object.get_surface_coordinates(np.array(self.coordinates_baseline["\"u\""]), np.array(self.coordinates_baseline["\"v\""]))
        x_array = profile_blade[0,:].real
        y_array = profile_blade[1,:].real
        
        return np.column_stack((x_array, y_array))
    
    def write_ffd(self, profile: np.ndarray, Delta: np.ndarray, outdir: str, gid: int = 0, cid: int = 0) -> str:
        """
        **Writes** the deformed geometry to file and **returns** /path/to/outdir/outfile.

        - profile (np.ndarray): the deformed geometry coordinates to be written to outfile.
        - Delta (np.ndarray): the deformation vector.
        - outdir (str): the output directory (it is to be combined with outfile).
        """

        outfile = f"{self.dat_file.split('/')[-1][:-4]}_g{gid}_c{cid}.dat"
        check_dir(outdir)
        logger.info(f"write profile g{gid} c{cid} as {outfile} to {outdir}")
        np.savetxt(os.path.join(outdir, outfile), profile[:,:],
                   header=f"Deformed profile {outfile}\nParams={[d for d in Delta]}")
        return os.path.join(outdir, outfile)

class ParaBlade_2D(ParaBlade):
    def __init__(self, dat_file: str, baseline_dir: str, param_bounds: dict, header: int = 2, scale: float = 1, **kwargs):

        super().__init__(dat_file, baseline_dir, param_bounds, header, scale, **kwargs)

    def apply_ffd(self, Delta: np.ndarray, dirNumber : int = None, no_rewriting: bool = False) -> np.ndarray:
                  
        dir = os.path.join(self.baseline_dir, 'temp'+str(dirNumber))
        params_dict = self.generate_dict(Delta)

        self.write_config_input(dir, params_dict, no_rewriting)

        profile = self.extract_data(dir)

        if dirNumber is None:
            os.system(f'rm -r {dir}')

        return profile*self.scale

class ParaBlade_POD_2D(ParaBlade):
    """
    Class to implement POD+Parablade_2D element. (Very similar to DLR_POD_2D class,
    small adjustments made to match ParaBlade implementation).
    """ 

    def __init__(self,
                 dat_file: str,
                 baseline_dir: str,
                 param_bounds: dict,
                 pod_ncontrol: int,
                 ffd_dataset_size: int,
                 seed: int = 123,
                 header: int = 2,
                 scale: float = 1,
                 perturb_POD = 'TrueMean', **kwargs):

        super().__init__(dat_file, baseline_dir, param_bounds, header, scale, **kwargs)

        self.ffd_ncontrol = len(self.param_bounds)
        self.pod_ncontrol = pod_ncontrol
        self.dat_file = dat_file
        self.perturb_POD = perturb_POD

        self.ffd_dataset_size = ffd_dataset_size
        self.ffd = ParaBlade_2D(dat_file, baseline_dir, param_bounds, scale=scale)
        self.seed = seed
        self.build_pod_dataset()
    
    def build_pod_dataset(self):
        sampler = qmc.LatinHypercube(d=self.ffd_ncontrol, seed=self.seed)
        sample = sampler.random(n=self.ffd_dataset_size)

        keys = list(self.param_bounds.keys())
        ul_bounds = np.array(list(self.param_bounds.values()))
        scaled_sample = qmc.scale(sample, l_bounds=ul_bounds[:,0], u_bounds=ul_bounds[:,1])

        orig_profiles = Parallel(n_jobs=96)(delayed(self.ffd.apply_ffd)
                                   (scaled_sample[i],i, True)
                                   for i in range(len(scaled_sample)))

        profiles = []
        for i, p in enumerate(orig_profiles):
            profiles.append(p)
            shutil.rmtree(os.path.join(self.baseline_dir, 'temp'+str(i)))
        self.profiles = copy.deepcopy(profiles)
        
        if self.perturb_POD == 'Baseline':
            for idx in range(len(profiles)):
                profiles[idx][:,1] = self.profiles[idx][:,1] - self.ffd.pts[:,1]


        self.S = np.stack([p[:, -1] for p in profiles] , axis=1)
        self.S_mean = 1 / len(profiles) * np.sum(self.S, axis=1)
        # self.S_mean = self.ffd.pts[:, -1]
        self.F = self.S[:, :] - self.S_mean[:, None]
        self.C = np.matmul(np.transpose(self.F), self.F)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.C)
        self.phi = np.matmul(self.F, self.eigenvectors)

        nmode = self.pod_ncontrol
        self.phi_tilde = self.phi[:, -nmode:]
        self.V_tilde_inv = np.linalg.inv(self.eigenvectors)[-nmode:, :]
        self.D_tilde = self.S_mean[:, None] + np.matmul(self.phi_tilde, self.V_tilde_inv)

        logger.info(f"POD Bounds are : {self.get_bound()}")

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        if self.perturb_POD == 'TrueMean':
            output_profile = np.column_stack((self.ffd.pts[:, 0], self.S_mean + np.sum(self.phi_tilde * Delta, axis=1)))
        elif self.perturb_POD == 'Baseline':
            output_profile = np.column_stack((self.ffd.pts[:, 0], self.S_mean + self.ffd.pts[:, 1] + np.sum(self.phi_tilde * Delta, axis=1)))
        return output_profile

    def get_bound(self) -> tuple[list[float], list[float]]:
        l_bound = [min(v) for v in self.V_tilde_inv]
        u_bound = [max(v) for v in self.V_tilde_inv]
        return l_bound, u_bound


class DLR_2D(DLR):

    def __init__(self, dat_file: str, home_dir: str, bladegen_path: str, param_bounds: dict,  header: int = 2, scale: float = 1, **kwargs):

        super().__init__(dat_file, home_dir, bladegen_path, param_bounds, header, scale, **kwargs)

    def apply_ffd(self, Delta: np.ndarray, dirNumber : int = None, no_rewriting: bool = False) -> np.ndarray:

        dir = os.path.join(self.home_dir, 'temp'+str(dirNumber))
        params_dict = self.generate_dict(Delta)

        self.write_progen_input(dir, params_dict, no_rewriting)

        self.run_blade_generator(dir)
        profile = self.extract_data_from_tec(os.path.join(dir, 'Output', 'Profile_mergedGestaf.tec'))
        if dirNumber is None:
            os.system(f'rm -r {dir}')

        return profile*self.scale

class DLR_POD_2D(DLR):

    def __init__(self,
                 dat_file: str,
                 home_dir: str,
                 bladegen_path: str,
                 param_bounds: dict,
                 pod_ncontrol: int,
                 ffd_dataset_size: int,
                 seed: int = 123,
                 header: int = 2,
                 scale: float = 1,
                 perturb_POD = 'TrueMean', **kwargs):

        super().__init__(dat_file, home_dir, bladegen_path, param_bounds, header, scale, **kwargs)

        self.ffd_ncontrol = len(self.param_bounds)
        self.pod_ncontrol = pod_ncontrol
        self.dat_file = dat_file
        self.perturb_POD = perturb_POD

        self.ffd_dataset_size = ffd_dataset_size
        self.ffd = DLR_2D(dat_file, home_dir, bladegen_path, param_bounds, scale=scale)
        self.seed = seed
        self.build_pod_dataset()

    def build_pod_dataset(self):
        sampler = qmc.LatinHypercube(d=self.ffd_ncontrol, seed=self.seed)
        sample = sampler.random(n=self.ffd_dataset_size)

        keys = list(self.param_bounds.keys())
        ul_bounds = np.array(list(self.param_bounds.values()))
        scaled_sample = qmc.scale(sample, l_bounds=ul_bounds[:,0], u_bounds=ul_bounds[:,1])

        orig_profiles = Parallel(n_jobs=96)(delayed(self.ffd.apply_ffd)
                                   (scaled_sample[i],i, True)
                                   for i in range(len(scaled_sample)))

        profiles = []
        for p in orig_profiles:
            profiles.append(self.interpolate_profile(p, self.pts, kind='linear'))
        profiles.append(self.interpolate_profile(self.pts, self.pts, kind='linear'))
        # print(self.interpolate_profile(p, self.pts, kind='linear'))
        # print(self.interpolate_profile(self.pts, self.pts, kind='linear'))
        # input()
        self.profiles = copy.deepcopy(profiles)
        
        if self.perturb_POD == 'Baseline':
            for idx in range(len(profiles)):
                profiles[idx][:,1] = self.profiles[idx][:,1] - self.ffd.pts[:,1]


        self.S = np.stack([p[:, -1] for p in profiles] , axis=1)
        self.S_mean = 1 / len(profiles) * np.sum(self.S, axis=1)
        # self.S_mean = self.ffd.pts[:, -1]
        self.F = self.S[:, :] - self.S_mean[:, None]
        self.C = np.matmul(np.transpose(self.F), self.F)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.C)
        self.phi = np.matmul(self.F, self.eigenvectors)

        nmode = self.pod_ncontrol
        self.phi_tilde = self.phi[:, -nmode:]
        self.V_tilde_inv = np.linalg.inv(self.eigenvectors)[-nmode:, :]
        self.D_tilde = self.S_mean[:, None] + np.matmul(self.phi_tilde, self.V_tilde_inv)

        logger.info(f"POD Bounds are : {self.get_bound()}")

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        if self.perturb_POD == 'TrueMean':
            output_profile = np.column_stack((self.ffd.pts[:, 0], self.S_mean + np.sum(self.phi_tilde * Delta, axis=1)))
        elif self.perturb_POD == 'Baseline':
            output_profile = np.column_stack((self.ffd.pts[:, 0], self.S_mean + self.ffd.pts[:, 1] + np.sum(self.phi_tilde * Delta, axis=1)))
        return output_profile

    def get_bound(self) -> tuple[list[float], list[float]]:
        l_bound = [min(v) for v in self.V_tilde_inv]
        u_bound = [max(v) for v in self.V_tilde_inv]
        return l_bound, u_bound

class FFD_2D(Deform):
    """
    This class implements a simple 2D FFD algorithm with deformation /y only.

    For ncontrol = 2 i.e. 2 control points per side, the unperturbed lattice is:

            P01 ----- P11 ---- P21 ---- P31
              |                         |
              |     ***************     |
              |    **** profile ****    |
              |     ***************     |
              |                         |
            P00 ----- P10 ---- P20 ---- P30

    with (P00, P30, P01, P31) fixed if pad = (1, 1).
    """
    def __init__(
            self, dat_file: str, ncontrol: int,
            pad: tuple[int, int] = (1, 1), **kwargs
    ):
        """
        Instantiates the FFD_2D object.

        **Input**

        - dat_file (str): path to input_geometry.dat.
        - ncontrol (int): the number of control points on each side of the lattice.
        - pad (tuple[int, int]): padding around the displacement vector.

        **Inner**

        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.

        - L (int): the number of control points in the x direction of each side of the lattice.
        - M (int): the number of control points in the y direction of each side of the lattice.
        - lat_pts (np.ndarray): the geometry coordinates in the lattice referential.
        """
        super().__init__(dat_file, ncontrol, **kwargs)
        assert pad in [(0, 0), (1, 1), (0, 1), (1, 0)], f"wrong padding: {pad}"
        self.pad: tuple[int, int] = pad
        self.L: int = ncontrol - 1 + sum(pad)
        self.M: int = 1
        self.build_lattice()
        self.lat_pts: np.ndarray = self.to_lat(self.pts)

    def build_lattice(self):
        """
        **Builds** a rectangle lattice with x1 as its origin.
        """
        epsilon = 0.
        self.min_x = np.min(self.pts, axis=0)[0] - epsilon
        self.max_x = np.max(self.pts, axis=0)[0] + epsilon
        self.min_y = np.min(self.pts, axis=0)[1] - epsilon
        self.max_y = np.max(self.pts, axis=0)[1] + epsilon
        self.x1 = np.array([self.min_x, self.min_y])

    def to_lat(self, pts: np.ndarray) -> np.ndarray:
        """
        **Returns** the coordinates projected in the lattices referential.

        - pts (np.ndarray): the geometry coordinates in the original referential.
        """
        if len(pts.shape) == 1:
            return np.array([(pts[0] - self.min_x) / (self.max_x - self.min_x),
                             (pts[1] - self.min_y) / (self.max_y - self.min_y)])
        return np.column_stack(((pts[:, 0] - self.min_x) / (self.max_x - self.min_x),
                                (pts[:, 1] - self.min_y) / (self.max_y - self.min_y)))

    def from_lat(self, pts: np.ndarray) -> np.ndarray:
        """
        **Returns** lattice coordinates back in the original referential.
        """
        if len(pts.shape) == 1:
            return np.array([pts[0] * (self.max_x - self.min_x) + self.min_x,
                             pts[1] * (self.max_y - self.min_y) + self.min_y])
        return np.column_stack((pts[:, 0] * (self.max_x - self.min_x) + self.min_x,
                                pts[:, 1] * (self.max_y - self.min_y) + self.min_y))

    def dPij(self, i: int, j: int, Delta: np.ndarray) -> np.ndarray:
        """
        **Returns** y-oriented displacement coordinates dPij from a 1D array Delta.
        """
        return np.array([0., Delta[i + j * (self.L + 1)]])

    def pad_Delta(self, Delta: np.ndarray) -> np.ndarray:
        """
        **Returns** padded Delta = [0, dP10, dP20, ..., dP{nc}0, 0, 0, dP11, dP21, ..., dP{nc}1, 0]
        with nc = ncontrol.

        - Delta (np.ndarray): the non-padded deformation vector.
        """
        return np.concatenate((np.pad(Delta[:self.ncontrol], self.pad),
                               np.pad(Delta[self.ncontrol:], self.pad)))

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        """
        **Returns** a new profile resulting from a perturbation Delta in the original referential.

        - Delta (np.ndarray): the deformation vector.</br>
          Delta = [dP10, dP20, ..., dP{nc}0, dP11, dP21, ..., dP{nc}1] with nc = ncontrol.
        """
        assert len(Delta) == 2 * self.ncontrol, f"len(Delta) {len(Delta)} != {2 * self.ncontrol}"
        Delta = self.pad_Delta(Delta)
        new_profile = []
        for x in self.lat_pts:
            x_new = x.copy()
            for ll in range(self.L + 1):
                for m in range(self.M + 1):
                    x_new += (math.comb(self.L, ll) * (1 - x[0])**(self.L - ll)
                              * math.comb(self.M, m) * (1 - x[1])**(self.M - m)
                              * x[0]**ll * x[1]**m * self.dPij(ll, m, Delta))
            new_profile.append([x_new])
        return self.from_lat(np.reshape(new_profile, (-1, 2)))

class FFD_POD_2D(Deform):
    """
    This class implements a 2D FFD-POD coupled class.
    """
    def __init__(
            self,
            dat_file: str,
            pod_ncontrol: int,
            ffd_ncontrol: int,
            ffd_dataset_size: int,
            ffd_bound: tuple[Any],
            seed: int = 123,
            **kwargs
    ):
        """
        Instantiates the FFD_POD_2D object.

        **Input**

        - dat_file (str): path to input_geometry.dat.
        - pod_ncontrol (int): the number of POD control points.
        - ffd_ncontrol (int): the number of FFD control points.
        - ffd_dataset_size (int): the number of ffd profiles in the POD dataset.
        - ffd_bound (tuple[Any]): the ffd dataset deformation boundaries.
        - seed (int): seed for the POD dataset sampling.
        - kwargs (dict): additional options to be passed to the FFD_2D inner object.

        **Inner**

        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.

        - ffd (FFD_2D): the ffd object used to build the POD dataset.
        """
        super().__init__(dat_file, ffd_ncontrol, **kwargs)
        self.pod_ncontrol = pod_ncontrol
        self.ffd_ncontrol = ffd_ncontrol
        self.ffd_dataset_size = ffd_dataset_size
        self.ffd = FFD_2D(dat_file, ffd_ncontrol // 2, **kwargs)
        self.ffd_bound = ffd_bound
        self.seed = seed
        self.build_pod_dataset()

    def build_pod_dataset(self):
        sampler = qmc.LatinHypercube(d=self.ffd_ncontrol, seed=self.seed)
        sample = sampler.random(n=self.ffd_dataset_size)
        scaled_sample = qmc.scale(sample, *self.ffd_bound)

        profiles = []
        for Delta in scaled_sample:
            profiles.append(self.ffd.apply_ffd(Delta))

        self.S = np.stack([p[:, -1] for p in profiles] , axis=1)
        self.S_mean = 1 / len(profiles) * np.sum(self.S, axis=1)
        self.F = self.S[:, :] - self.S_mean[:, None]
        self.C = np.matmul(np.transpose(self.F), self.F)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.C)
        self.phi = np.matmul(self.F, self.eigenvectors)

        nmode = self.pod_ncontrol
        self.phi_tilde = self.phi[:, -nmode:]
        self.V_tilde_inv = np.linalg.inv(self.eigenvectors)[-nmode:, :]
        self.D_tilde = self.S_mean[:, None] + np.matmul(self.phi_tilde, self.V_tilde_inv)

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (self.ffd.pts[:, 0], self.S_mean + np.sum(self.phi_tilde * Delta, axis=1))
        )

    def get_bound(self) -> tuple[list[float], list[float]]:
        l_bound = [min(v) for v in self.V_tilde_inv]
        u_bound = [max(v) for v in self.V_tilde_inv]
        return l_bound, u_bound

class RotationWrapper(Deform):
    def __init__(self, deform_obj: Deform):
        self._deform_obj = deform_obj

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        theta_rad = Delta[-1] / 180. * np.pi
        rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                               [np.sin(theta_rad), np.cos(theta_rad)]])
        profile = self._deform_obj.apply_ffd(Delta[:-1])
        cog = get_cog(profile)
        return (profile - cog) @ rot_matrix.T + cog

    def __getattr__(self, name):
        return getattr(self._deform_obj, name)
