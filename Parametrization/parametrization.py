#### package import ####

import numpy as np
import tensorflow as tf
import torch
from Parametrization.optimizers import Optimizer
from System.system import *

class Parametrization:
    """
    Parametrizes your molecular system.

    Parameters
    ----------
    molecular_system : system.FMolecular_system object
        Contains properties of the molecular system that is to be parametrized
    term_type : str
        'energy' or 'force' or 'force & energy'; determines which terms go into the objective function
    optimizer : Parametrization.optimiizers.Optimizer object
        Contains optimizer type and specific settings
    regularization : bool
        whether a penalty term is used or not

    parameters : 1d array
        contains optimized, scaled & vectorized parameters
    """
    def __init__(self, molecular_system=None, term_type=None, optimizer=None, regularization=False):

        assert molecular_system is not None, 'no molecular system selected'

        self.molecular_system = molecular_system
        self.emm = self.molecular_system.mm_energies['all']
        self.fmm = self.molecular_system.mm_net_forces
        self.eqm = self.molecular_system.qm_energies['all']
        self.fqm = self.molecular_system.qm_net_forces
        self.weights = self.molecular_system.weights
        self.n_atoms = self.molecular_system.n_atoms
        self.parameters = None

        self.optimizer = optimizer
        self.regularization = regularization

        self.term_types = ['energy', 'force', 'force&energy']
        self.term_type = term_type

        self.step = 0

    def calculate_classical_energies_forces(self):
        """
        calculates all classical energies and forces using OpenMM for the molecular_system

        sets :
            self.emm
        """

        for omm_system_name in self.molecular_system.openmm_systems.keys():
            if self.molecular_system.openmm_systems[omm_system_name] != None:
                self.molecular_system.generate_mm_energies_forces(omm_system_name)

        self.emm = self.molecular_system.mm_energies['all']

    def calculate_classical_net_forces(self):
        """
        calculates classical net forces (if more than one molecule/ molecule is solvated)

        sets :
            self.fmm
        """

        self.molecular_system.calculate_mm_net_forces()

        self.fmm = self.molecular_system.mm_net_forces

    def evaluate_obj_func_energy(self): 
        """
        calculates the value of the objective function using the format required by the selected optimizer
            n_conf        (ΔE - <ΔE>)²
        O =   Σ  ω_{conf} ------------
             conf         Var(E^{QM})

        returns :
            value of the objective function in the format of the selected optimizer
        """

        for sys_type in self.molecular_system.openmm_systems.keys():

            if self.molecular_system.openmm_systems[sys_type] != None:

                self.molecular_system.openmm_systems[sys_type].set_parameters()

        if isinstance(self.weights, np.ndarray) == False:

            self.molecular_system.generate_weights()
            self.weights = self.molecular_system.weights
            weights = np.atleast_2d(self.weights) #weights weigh conformations
            weights = np.atleast_3d(weights.T)

        else: 

            weights = np.atleast_2d(self.weights) #weights weigh conformations
            weights = np.atleast_3d(weights.T)

        self.calculate_classical_energies_forces()

        if self.optimizer.opt_method == ("scipy" or "pso"):

            delta_E = self.emm - self.eqm
            enumerator = np.power((delta_E - np.mean(delta_E)), 2)
            denominator = np.var(self.eqm)
            frac = enumerator / denominator
            obj_f_e = np.sum(self.weights * frac)
            # n_conf        (ΔE - <ΔE>)²
            #   Σ  ω_{conf} ------------
            #  conf         Var(E^{QM})

        elif self.optimizer.opt_method == "tf_adam":

            tf_emm = tf.Variable(self.emm, dtype=float)
            tf_eqm = tf.Variable(self.eqm, dtype=float)
            tf_weights = tf.constant(self.weights, dtype=float)
            delta_E = tf.math.subtract(tf_emm, tf_eqm)
            enumerator = tf.math.pow((tf.math.subtract(delta_E, tf.math.reduce_mean(delta_E)), 2))
            denominator = tf.math.reduce_variance(tf_eqm)
            frac = tf.math.divide(enumerator, denominator)
            obj_f_e = tf.math.reduce_sum(tf.math.multiply(tf_weights, frac))
            # n_conf        (ΔE - <ΔE>)²
            #   Σ  ω_{conf} ------------
            #  conf         Var(E^{QM})

        elif self.optimizer.opt_method == ("pt_adam" or "pt_lbfgs"):

            pt_emm = torch.tensor(self.emm, dtype=torch.float64, requires_grad=True)
            pt_eqm = torch.tensor(self.eqm, dtype=torch.float64, requires_grad=True)
            pt_weights = torch.tensor(self.weights, dtype=torch.float64, requires_grad=False)
            delta_E = pt_emm - pt_eqm
            enumerator = torch.pow((delta_E - torch.mean(delta_E)), 2)
            denominator = torch.var(pt_eqm)
            frac = torch.div(enumerator, denominator)
            obj_f_e = torch.sum(torch.mul(pt_weights, frac))
            # n_conf        (ΔE - <ΔE>)²
            #   Σ  ω_{conf} ------------
            #  conf         Var(E^{QM})

        else:
            print('optimizer of type {} not implemented'.format(Optimizer.opt_method.lower()))

        return obj_f_e

    def evaluate_obj_func_force(self, method=None): 
        """
        calculates the value of the objective function using the format required by the selected optimizer
                 1            n_atoms n_conf              |ΔF|²
        O = ----------------     Σ     Σ     ω_{conf} ------------
            3n_atoms n_confs   atom   conf             Var(F^{QM})

        returns :
            value of the objective function in the format of the selected optimizer
        """

        assert method in [None, "variance", "covariance"], "Force property term for method {} is not implemented.".format(method)

        for sys_type in self.molecular_system.openmm_systems.keys():

            if self.molecular_system.openmm_systems[sys_type] != None:

                self.molecular_system.openmm_systems[sys_type].set_parameters()

        if isinstance(self.weights, np.ndarray) == False:

            self.molecular_system.generate_weights()
            self.weights = self.molecular_system.weights
            weights = np.atleast_2d(self.weights) #weights weigh conformations
            weights = np.atleast_3d(weights.T)

        else: 

            weights = np.atleast_2d(self.weights) #weights weigh conformations
            weights = np.atleast_3d(weights.T)

        self.calculate_classical_energies_forces
        self.calculate_classical_net_forces

        if self.optimizer.opt_method == ("scipy" or "pso"):

            delta_F = self.fmm - self.fqm

            if method is None:
                method = "variance"

            if method == "variance":

                enumerator = np.power(np.abs(delta_F), 2)
                denominator = np.var(np.linalg.norm(self.fqm, axis=2))
                frac_weighted = weights * (enumerator / denominator)
                obj_f_f = np.sum(frac_weighted) / (3 * self.n_atoms['all'])
                #     1            n_atoms n_conf              |ΔF|²
                # ----------------     Σ     Σ     ω_{conf} ------------
                # 3n_atoms n_confs   atom   conf             Var(F^{QM})

        elif self.optimizer.opt_method == "tf_adam":

            tf_fmm = tf.Variable(self.fmm, dtype=float)
            tf_fqm = tf.Variable(self.fqm, dtype=float)
            tf_weights = tf.constant(weights, dtype=float)
            tf_n_atoms = tf.constant(self.n_atoms['all'], dtype=int)
            delta_F = tf.math.subtract(tf_fmm, tf_fqm)

            if method is None:
                method = "variance"

            if method == "variance":

                enumerator = tf.math.pow(tf.math.abs(delta_F), 2)
                denominator = tf.math.reduce_variance(tf.linalg.normalize(tf_fqm, axis=2))
                frac_weighted = tf.math.multiply(tf_weights, (tf.math.divide(enumerator, denominator)))
                obj_f_f = tf.math.divide(tf.math.reduce_sum(frac_weighted), tf.math.multiply(3, tf_n_atoms))
                #     1            n_atoms n_conf              |ΔF|²
                # ----------------     Σ     Σ     ω_{conf} ------------
                # 3n_atoms n_confs   atom   conf             Var(F^{QM})

        elif self.optimizer.opt_method == ("pt_adam" or "pt_lbfgs"):
              
            pt_fmm = torch.tensor(self.fmm, dtype=torch.float64, requires_grad=True)
            pt_fqm = torch.tensor(self.fqm, dtype=torch.float64, requires_grad=True)
            pt_weights = torch.tensor(weights, dtype=torch.float64, requires_grad=False)
            pt_n_atoms = torch.tensor(self.n_atoms['all'], dtype=torch.int64, requires_grad=False)
            pt_f = tf_weights ("obj_f_f")
            delta_F = pt_fmm - pt_f
            delta_F = pt_fmm - pt_fqm

            if method is None:
                method = "variance"

            if method == "variance":
                  
                enumerator = torch.pow(torch.abs(delta_F), 2)
                denominator = torch.var(torch.linalg.norm(pt_fqm, axis=2))
                frac_weighted = torch.mul(pt_weights, torch.div(enumerator, denominator))
                obj_f_f = torch.sum(torch.div(frac_weighted, torch.mul(3, pt_n_atoms)))
                #     1            n_atoms n_conf              |ΔF|²
                # ----------------     Σ     Σ     ω_{conf} ------------
                # 3n_atoms n_confs   atom   conf             Var(F^{QM})

        else:
            print('optimizer of type {} not implemented'.format(Optimizer.opt_method.lower()))
        
        return obj_f_f
    

    def calc_force_std_dev(self):
        """
        calculates the standard deviation of forces

        """
        delta_F = self.fmm - self.fqm
        force_std_dev = np.std(delta_F)

        return force_std_dev
    
    
    def wrap_objective_function(self, parameters): #TODO: has to be abstract function of parameters
        """
        Abstract function that calculates the value of the objective function based on the parameters

        Parameters
        ----------
        parameters : 1d array
            Selected scaled and vectorized parameters from molecular_system.scaled_parameters
        other (internal) parameters:
            self.molecular_system : system.Molecular_system object
            self.term_type : str

        returns :
            value of objective funtion as float
        """
        self.molecular_system.scaled_parameters = parameters

        self.molecular_system.unscale_parameters()
        self.molecular_system.redistribute_vectorized_parameters()
        self.molecular_system.reshape_vectorized_parameters()
        self.molecular_system.expand_reduced_parameters()
            
        if self.term_type == 'energy':
            obj_f_value = self.evaluate_obj_func_energy()

        elif self.term_type == 'force':
            obj_f_value = self.evaluate_obj_func_force()

        elif self.term_type == 'force & energy':
            obj_f_f = self.evaluate_obj_func_force()
            obj_f_e = self.evaluate_obj_func_energy()
            obj_f_value = obj_f_f + obj_f_e

        self.molecular_system.reduce_ff_optimizable(self.molecular_system.slice_list)
        self.molecular_system.vectorize_reduced_parameters()
        self.molecular_system.merge_vectorized_parameters()
        self.molecular_system.scale_parameters()

        self.molecular_system.scaled_parameters = parameters


        #TODO: add regularization term if needed
        """
        if self.regularization:

            regularization_term = self.calculate_regularization_term()
            obj_f += regularization_term
        """

        return obj_f_value


    def parametrize(self, parameters, regularization_type='L2', scaling_factor=1.0, hyperbolic_beta=0.01):
        """
        Feeds the objective function wrapper and the parameters to the selected optimizer.

        Parameters:
        -----------
        parameters : 1d array
            Selected scaled and vectorized parameters from molecular_system.scaled_parameters
        
        sets: 
            self.parameters : 1d array
                optimized scaled & vectorized parameters
        """

        assert self.optimizer is not None, 'optimizer not set'
        assert self.term_type is not None, 'term_type not set'
        assert self.term_type in self.term_types, 'please choose a valid term_type'

        #TODO: implement regularization term if needed
        """
        if self.regularization:
            regularization_term = self.calculate_regularization_term(regularization_type, scaling_factor, hyperbolic_beta)
        else:
            regularization_term = 0.0
        """

        optimized_params, obj_f_value = self.optimizer.run_optimizer(self.wrap_objective_function, parameters)
        self.parameters = optimized_params

        #obj_f_value += regularization_term
