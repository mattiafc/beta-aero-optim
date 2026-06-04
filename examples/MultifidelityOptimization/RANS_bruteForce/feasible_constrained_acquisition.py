
import numpy as np
import logging
import pandas as pd

from typing import Callable

from scipy.stats            import norm
from scipy.spatial.distance import cdist

from pymoo.core.problem import Problem
from pymoo.optimize     import minimize
from pymoo.termination  import get_termination
from pymoo.core.result  import Result
from pymoo.algorithms.soo.nonconvex.pso import PSO

from aero_optim.dlr.dlr import FFD_2D
from aero_optim.mf_sm.mf_infill import (EDProblem, compute_pareto)
from aero_optim.mf_sm.mf_models import MfLGP, MfKPLS, MfSMT, SfSMT, MultiObjectiveModel
from aero_optim.geom import (get_area, get_camber_th, get_chords, get_circle, get_circle_centers,
                             get_cog, get_radius_violation, split_profile, plot_profile, plot_sides)

# Configurazione feasibility thresholds (modificabili manualmente)
FEASIBILITY_BOUNDS = {
    'ADP': [-0.67,0.93],  # theta_ADP bounds
    'OP1': [-0.22,2.78],  # theta_OP1 bounds
    'OP2': [-2.67,0.33]   # theta_OP2 bounds
}

logger = logging.getLogger(__name__)


def compute_QoIs(profile, verbose = False):
    # LE and TE defined erroneously as min and max x-coord
    c, c_ax = get_chords(profile)
    upper, lower = split_profile(profile)

    camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
    th_over_c = thmax/c
    Xth_over_cax = Xthmax/c_ax

    area = abs(get_area(profile))
    area_over_c2 = area/c**2
    cog = get_cog(profile)
    Xcg_over_cax = cog[0] / c_ax
    
    return(th_over_c, Xth_over_cax, area_over_c2, Xcg_over_cax)

class AcquisitionFunctionProblem(Problem):
    """
    Generic class for Bayesian acquisition function optimization problems.
    """
    def __init__(
            self,
            function: Callable,
            model: MfLGP | MfKPLS | MfSMT | SfSMT | MultiObjectiveModel,
            ffd: FFD_2D,
            n_var: int,
            bound: list,
            min: bool = True
    ):
        Problem.__init__(
            self, n_var=n_var, n_ieq_constr=4, xl=bound[0], xu=bound[1]
        )
        self.model = model
        self.function = function
        self.min = min
        self.ffd = ffd
        self.gen_ctr = 0
        self.set_inner()


    def set_inner(self):
        """
        **Sets** some baseline quantities required to compute the relative constraints:
        
        - bsl_camber_th (tuple[np.ndarray, float, float, np.ndarray])
        - bsl_area (float)
        - bsl_c (float)
        - bsl_c_ax (float)
        - bsl_cog (np.ndarray)
        - bsl_cog_x (float)
        - constraint (bool): whether to apply constraints (True) or not (False).
        """
        
        bsl_pts = self.ffd.pts
        self.bsl_c, self.bsl_c_ax = get_chords(bsl_pts)
        logger.info(f"baseline chord = {self.bsl_c} m, baseline axial chord = {self.bsl_c_ax}")
        bsl_upper, bsl_lower = split_profile(bsl_pts)

        self.bsl_camber_th = get_camber_th(bsl_upper, bsl_lower, interpolate=True)
        self.bsl_th_over_c = self.bsl_camber_th[1] / self.bsl_c
        self.bsl_Xth_over_cax = self.bsl_camber_th[2] / self.bsl_c_ax

        logger.info(f"baseline th_max = {self.bsl_camber_th[1]} m, "
                    f"Xth_max {self.bsl_camber_th[2]} m, "
                    f"th_max / c = {self.bsl_th_over_c}, "
                    f"Xth_max / c_ax = {self.bsl_Xth_over_cax}")

        self.bsl_area = abs(get_area(bsl_pts))
        self.bsl_area_over_c2 = self.bsl_area / self.bsl_c**2
        logger.info(f"baseline area = {self.bsl_area} m2, "
                    f"baseline area / (c * c) = {self.bsl_area_over_c2}")
        
        self.bsl_cog = get_cog(bsl_pts)
        self.bsl_Xcg_over_cax = self.bsl_cog[0] / self.bsl_c_ax
        logger.info(f"baseline X_cg over c_ax = {self.bsl_Xcg_over_cax}")
        
    def compute_constrained_candidates(self, X: np.ndarray) -> np.ndarray:

        th_constraint   = []
        Xth_constraint  = []
        area_constraint = []
        cog_constraint  = []
        cont = 0
        for x in X:
            profile = self.ffd.apply_ffd(x)
            # np.savetxt(f"Candidates/profile_g{self.gen_ctr}_c{cont}.dat", profile)
            th_over_c, Xth_over_cax, area_over_c2, Xcg_over_cax = compute_QoIs(profile)

            th_constraint.append(abs(th_over_c - self.bsl_th_over_c) / self.bsl_th_over_c - 0.3)
            Xth_constraint.append(abs(Xth_over_cax - self.bsl_Xth_over_cax) / self.bsl_Xth_over_cax - 0.2)
            area_constraint.append(abs(area_over_c2 - self.bsl_area_over_c2) / self.bsl_area_over_c2 - 0.2)
            cog_constraint.append(abs(Xcg_over_cax - self.bsl_Xcg_over_cax) / self.bsl_Xcg_over_cax - 0.2)
            cont+=1

        return np.column_stack([th_constraint, Xth_constraint, area_constraint, cog_constraint])

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        out["G"] = self.compute_constrained_candidates(X)
        out["F"] = self.function(X, self.model) if self.min else -self.function(X, self.model)
        nViolating = np.sum(np.any(np.array(out["G"]) > 0, axis=1))
        if nViolating>0:
            print(f'{nViolating} profiles violating constraints in generation {self.gen_ctr}')
        self.gen_ctr += 1


def probability_feasibility(x: np.ndarray, model: MultiObjectiveModel) -> np.ndarray:
    """
    Computes the probability of feasibility for a Gaussian variable.
    P(u <= threshold) = 1 - P(u > threshold) = 1 - Phi((u - threshold) / std)
    where Phi is the CDF of the standard normal distribution.
    """

    u_beta_ADP = model.models[2].evaluate(x)
    u_beta_OP1 = model.models[3].evaluate(x)
    u_beta_OP2 = model.models[4].evaluate(x)

    std_beta_ADP = model.models[2].evaluate_std(x) + 1e-6
    std_beta_OP1 = model.models[3].evaluate_std(x) + 1e-6
    std_beta_OP2 = model.models[4].evaluate_std(x) + 1e-6

    prob_feas_ADP = norm.cdf((FEASIBILITY_BOUNDS["ADP"][1]-u_beta_ADP)/std_beta_ADP) - norm.cdf((FEASIBILITY_BOUNDS["ADP"][0]-u_beta_ADP)/std_beta_ADP)
    prob_feas_OP1 = norm.cdf((FEASIBILITY_BOUNDS["OP1"][1]-u_beta_OP1)/std_beta_OP1) - norm.cdf((FEASIBILITY_BOUNDS["OP1"][0]-u_beta_OP1)/std_beta_OP1)
    prob_feas_OP2 = norm.cdf((FEASIBILITY_BOUNDS["OP2"][1]-u_beta_OP2)/std_beta_OP2) - norm.cdf((FEASIBILITY_BOUNDS["OP2"][0]-u_beta_OP2)/std_beta_OP2)

    return (prob_feas_ADP*prob_feas_OP1*prob_feas_OP2).flatten()

def MPI_acquisition_function_constrained(x: np.ndarray, model: MultiObjectiveModel) -> np.ndarray:
    """
    Bi-objective Minimal Probability of Improvement:
    see A. A. Rahat (2017): https://dl.acm.org/doi/10.1145/3071178.3071276
    """
    assert model.models[0].y_hf_DOE is not None
    assert model.models[1].y_hf_DOE is not None
    pareto = compute_pareto(model.models[0].y_hf_DOE, model.models[1].y_hf_DOE)

    u1 = model.models[0].evaluate(x)
    u2 = model.models[1].evaluate(x)
    std1 = model.models[0].evaluate_std(x) + 1e-6
    std2 = model.models[1].evaluate_std(x) + 1e-6

    MPI = np.min(np.column_stack(
        [1 - norm.cdf((u1 - pp[0]) / std1) * norm.cdf((u2 - pp[1]) / std2)
            for pp in pareto]
    ), axis=1)
    
    return MPI*probability_feasibility(x, model)

def maximize_MPI_BO_constrained(
        model: MultiObjectiveModel, ffd : FFD_2D,
        n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Bi-objective Minimal Probability of Improvement maximization.
    """
    problem = AcquisitionFunctionProblem(MPI_acquisition_function_constrained, model, ffd, n_var, bound, min=False)

    point = optimize_acquisition_function(problem, seed, n_gen=n_gen).X
    print("MPI probability of feasibility: ", probability_feasibility(np.atleast_2d(point), model).item())
    print("MPI candidate point theta_ADP:  ", model.models[2].evaluate(np.atleast_2d(point)).item())
    print("MPI candidate point theta_OP1:  ", model.models[3].evaluate(np.atleast_2d(point)).item())
    print("MPI candidate point theta_OP2:  ", model.models[4].evaluate(np.atleast_2d(point)).item())
    
    return point

def LCB_acquisition_function_constrained(x: np.ndarray, model: MultiObjectiveModel, idx: int, alpha: float = 1) -> np.ndarray:
    """
    Lower Confidence Bound acquisition function.
    """
    return (model.models[idx].evaluate(x) - alpha * model.models[idx].evaluate_std(x)).flatten()*(1-probability_feasibility(x, model))

def minimize_LCB_constrained(
        model: MultiObjectiveModel, ffd : FFD_2D,
        idx: int,
        n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Lower Confidence Bound minimization function.
    """
    print(idx)
    lcb_func = lambda x,_: LCB_acquisition_function_constrained(x, model, idx)
    problem = AcquisitionFunctionProblem(lcb_func, model, ffd, n_var, bound)

    point = optimize_acquisition_function(problem, seed, n_gen=n_gen).X

    print("LCB probability of feasibility: ", probability_feasibility(np.atleast_2d(point), model).item())
    print("LCB candidate point theta_ADP:  ", model.models[2].evaluate(np.atleast_2d(point)).item())
    print("LCB candidate point theta_OP1:  ", model.models[3].evaluate(np.atleast_2d(point)).item())
    print("LCB candidate point theta_OP2:  ", model.models[4].evaluate(np.atleast_2d(point)).item())
    
    return point

def ED_acquisition_function_constrained(x: np.ndarray, DOE: np.ndarray) -> np.ndarray:
    """
    Euclidean Distance:
    see X. Zhang et al. (2021): https://doi.org/10.1016/j.cma.2020.113485
    """
    f1 = np.min(cdist(np.atleast_2d(x), DOE, "euclidean"), axis=1)
    f1 = np.expand_dims(f1, axis=1)
    assert f1.shape == (x.shape[0], 1), f"f1 shape {f1.shape} x shape {x.shape}"
    return f1
 
class EDProblem_constrained(Problem):
    
    """
    Euclidean Distance problem.
    """
    def __init__(self, DOE: np.ndarray, ffd: FFD_2D, n_var: int, bound: list):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=4, xl=bound[0], xu=bound[-1])
        self.DOE = DOE
        self.ffd = ffd
        self.gen_ctr = 0
        self.set_inner()


    def set_inner(self):
        """
        **Sets** some baseline quantities required to compute the relative constraints:
        
        - bsl_camber_th (tuple[np.ndarray, float, float, np.ndarray])
        - bsl_area (float)
        - bsl_c (float)
        - bsl_c_ax (float)
        - bsl_cog (np.ndarray)
        - bsl_cog_x (float)
        - constraint (bool): whether to apply constraints (True) or not (False).
        """
        
        bsl_pts = self.ffd.pts
        self.bsl_c, self.bsl_c_ax = get_chords(bsl_pts)
        logger.info(f"baseline chord = {self.bsl_c} m, baseline axial chord = {self.bsl_c_ax}")
        bsl_upper, bsl_lower = split_profile(bsl_pts)

        self.bsl_camber_th = get_camber_th(bsl_upper, bsl_lower, interpolate=True)
        self.bsl_th_over_c = self.bsl_camber_th[1] / self.bsl_c
        self.bsl_Xth_over_cax = self.bsl_camber_th[2] / self.bsl_c_ax

        logger.info(f"baseline th_max = {self.bsl_camber_th[1]} m, "
                    f"Xth_max {self.bsl_camber_th[2]} m, "
                    f"th_max / c = {self.bsl_th_over_c}, "
                    f"Xth_max / c_ax = {self.bsl_Xth_over_cax}")

        self.bsl_area = abs(get_area(bsl_pts))
        self.bsl_area_over_c2 = self.bsl_area / self.bsl_c**2
        logger.info(f"baseline area = {self.bsl_area} m2, "
                    f"baseline area / (c * c) = {self.bsl_area_over_c2}")
        
        self.bsl_cog = get_cog(bsl_pts)
        self.bsl_Xcg_over_cax = self.bsl_cog[0] / self.bsl_c_ax
        logger.info(f"baseline X_cg over c_ax = {self.bsl_Xcg_over_cax}")
        
    def compute_constrained_candidates(self, X: np.ndarray) -> np.ndarray:

        th_constraint   = []
        Xth_constraint  = []
        area_constraint = []
        cog_constraint  = []
        cont = 0
        for x in X:
            profile = self.ffd.apply_ffd(x)
            # np.savetxt(f"Candidates/profile_g{self.gen_ctr}_c{cont}.dat", profile)
            th_over_c, Xth_over_cax, area_over_c2, Xcg_over_cax = compute_QoIs(profile)

            th_constraint.append(abs(th_over_c - self.bsl_th_over_c) / self.bsl_th_over_c - 0.3)
            Xth_constraint.append(abs(Xth_over_cax - self.bsl_Xth_over_cax) / self.bsl_Xth_over_cax - 0.2)
            area_constraint.append(abs(area_over_c2 - self.bsl_area_over_c2) / self.bsl_area_over_c2 - 0.2)
            cog_constraint.append(abs(Xcg_over_cax - self.bsl_Xcg_over_cax) / self.bsl_Xcg_over_cax - 0.2)
            cont += 1
            
        return np.column_stack([th_constraint, Xth_constraint, area_constraint, cog_constraint])

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        out["G"] = self.compute_constrained_candidates(x)
        out["F"] = -ED_acquisition_function_constrained(x, self.DOE)
        nViolating = np.sum(np.any(np.array(out["G"]) > 0, axis=1))
        if nViolating>0:
            print(f'{nViolating} profiles violating constraints in generation {self.gen_ctr}')
        self.gen_ctr += 1


def maximize_ED_constrained(
        DOE: np.ndarray, ffd: FFD_2D, n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Euclidean distance maximization function.

    Inputs:
        DOE (np.ndarray): the full low- and high-fidelity DOE.
    """
    problem = EDProblem_constrained(DOE, ffd, n_var=n_var, bound=bound)
    point = optimize_acquisition_function(problem, seed, n_gen=n_gen).X
    return point

def optimize_acquisition_function(
        problem: EDProblem | AcquisitionFunctionProblem,
        seed: int, n_gen: int = 100
) -> Result:
    """
    Generic function that optimizes any given acquisition function problem.
    """
    res = minimize(problem, PSO(pop_size=50),
                   termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    logger.info(f"adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res
