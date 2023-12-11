#### package import ####

import numpy as np
import tensorflow as tf
import torch
from Parametrization.optimizers import Optimizer

class Parametrization:
    """
    Parametrizes your molecular system.

    Parameters
    ----------
    molecular_system : Molecular_system object
        Contains properties of the molecular system that is to be parametrized
    term_type : str
        'energy' or 'force' or 'force & energy'; determines which terms go into the objective function
    optimizer : Parametrization.optimiizers.Optimizer object
        Contains optimizer type and specific settings
    regularization : bool
        whether a penalty term is used or not

    parameters : dict
        contains parameters from OMM_Interface.openmm.OpenMM_system.ff_optimizable, their
        scaling constants, and the scaled parameters

    """
    def __init__(self, molecular_system=None, term_type=None, optimizer=None, regularization=False):

        assert molecular_system is not None, 'no molecular system selected'

        self.molecular_system = molecular_system
        self.emm = self.molecular_system.mm_energies
        self.fmm = self.molecular_system.mm_forces
        self.eqm = self.molecular_system.qm_energies
        self.fqm = self.molecular_system.qm_forces
        self.weights = self.molecular_system.weights
        self.n_atoms = self.molecular_system.n_atoms
        self.ff_optimizable = self.molecular_system.openmm_sys.ff_optimizable

        self.optimizer = optimizer
        self.regularization = regularization

        self.energy_properties = None
        self.force_properties = None

        self.parameters = {
            'original_parameters': [],
            'scaling_constants': [],
            'scaled_parameters': [],
            'current_params': [],
            'current_scaling_constants': [],
            'current_scaled_parameters': []
        }

        self.term_types = ['energy', 'force', 'force&energy']
        self.term_type = term_type

        self.step = 0

    def calculate_net_forces(self):

        if self.molecular_system.system_type == '2_gas_phase':
            net_forces = self.molecular_system.
    def recalculate_mm_properties(self):
        #implementing this method to update molecular system properties
        for force_key, parameters in self.ff_optimizable.items():
            if force_key != 'NBException':
                force = self.molecular_system.openmm_sys.getForce(force_key)
                for param_name, values in parameters.items():
                    if param_name not in ['atom1', 'atom2', 'atom3', 'atom4', 'periodicity']:
                        for atom_index, value in enumerate(values):
                            force.setParameterByIndex(atom_index, param_name, value)
                            

        for force_key in self.ff_optimizable.keys():
            self.molecular_system.openmm_sys.set_parameters(force_key)

        self.molecular_system.get_mm_energies_forces()

        self.emm = self.molecular_system.mm_energies
        self.fmm = self.molecular_system.mm_forces

    def calculate_obj_func_energy(self):
        if self.optimizer.opt_method == ("scipy" or "pso"):
            delta_E = self.emm - self.eqm
            enumerator = np.power((delta_E - np.mean(delta_E)), 2)
            denominator = np.var(self.eqm)
            frac = enumerator / denominator
            obj_f_e = np.sum(self.weights * frac)
        elif self.optimizer.opt_method == "tf_adam":
            tf_emm = tf.Variable(self.emm, dtype=float)
            tf_eqm = tf.Variable(self.eqm, dtype=float)
            tf_weights = tf.constant(self.weights, dtype=float)
            delta_E = tf.math.subtract(tf_emm, tf_eqm)
            enumerator = tf.math.pow((tf.math.subtract(delta_E, tf.math.reduce_mean(delta_E)), 2))
            denominator = tf.math.reduce_variance(tf_eqm)
            frac = tf.math.divide(enumerator, denominator)
            obj_f_e = tf.math.reduce_sum(tf.math.multiply(tf_weights, frac))
        elif self.optimizer.opt_method == ("pt_adam" or "pt_lbfgs"):
            pt_emm = torch.tensor(self.emm, dtype=torch.float64, requires_grad=True)
            pt_eqm = torch.tensor(self.eqm, dtype=torch.float64, requires_grad=True)
            pt_weights = torch.tensor(self.weights, dtype=torch.float64, requires_grad=False)
            delta_E = pt_emm - pt_eqm
            enumerator = torch.pow((delta_E - torch.mean(delta_E)), 2)
            denominator = torch.var(pt_eqm)
            frac = torch.div(enumerator, denominator)
            obj_f_e = torch.sum(torch.mul(pt_weights, frac))
        else:
            print('optimizer of type {} not implemented'.format(Optimizer.opt_method.lower()))
        return obj_f_e

    def calculate_obj_func_force(self, method=None):
        assert method in [None, "variance", "covariance"], "Force property term for method {} is not implemented.".format(method)
        weights = np.atleast_2d(self.weights)
        weights = np.atleast_3d(weights.T)

        if self.optimizer.opt_method == ("scipy" or "pso"):
            delta_F = self.fmm - self.fqm
            if method is None:
                method = "variance"

            if method == "variance":
                enumerator = np.power(np.abs(delta_F), 2)
                denominator = np.var(np.linalg.norm(self.fqm, axis=2))
                frac_weighted = weights * (enumerator / denominator)
                obj_f_f = np.sum(frac_weighted) / (3 * self.n_atoms)
        elif self.optimizer.opt_method == "tf_adam":
            tf_fmm = tf.Variable(self.fmm, dtype=float)
            tf_fqm = tf.Variable(self.fqm, dtype=float)
            tf_weights = tf.constant(weights, dtype=float)
            tf_n_atoms = tf.constant(self.n_atoms, dtype=int)
            delta_F = tf.math.subtract(tf_fmm, tf_fqm)

            if method is None:
                method = "variance"

            if method == "variance":
                enumerator = tf.math.pow(tf.math.abs(delta_F), 2)
                denominator = tf.math.reduce_variance(tf.linalg.normalize(tf_fqm, axis=2))
                frac_weighted = tf.math.multiply(tf_weights, (tf.math.divide(enumerator, denominator)))
                obj_f_f = tf.math.divide(tf.math.reduce_sum(frac_weighted), tf.math.multiply(3, tf_n_atoms))
        elif self.optimizer.opt_method == ("pt_adam" or "pt_lbfgs"):
              pt_fmm = torch.tensor(self.fmm, dtype=torch.float64, requires_grad=True)
              pt_fqm = torch.tensor(self.fqm, dtype=torch.float64, requires_grad=True)
              pt_weights = torch.tensor(weights, dtype=torch.float64, requires_grad=False)
              pt_n_atoms = torch.tensor(self.n_atoms, dtype=torch.int64, requires_grad=False)
              pt_f= tf_weights ("obj_f_f")
              delta_F = pt_fmm - pt_f
              delta_F = pt_fmm - pt_fqm

            if method is None:
                method = "variance"

            if method == "variance":
                enumerator = torch.pow(torch.abs(delta_F), 2)
                denominator = torch.var(torch.linalg.norm(pt_fqm, axis=2))
                frac_weighted = torch.mul(pt_weights, torch.div(enumerator, denominator))
                obj_f_f = torch.sum(torch.div(frac_weighted, torch.mul(3, pt_n_atoms)))
        else:
            print('optimizer of type {} not implemented'.format(Optimizer.opt_method.lower()))
        
        return obj_f_f

    def calc_scaling_constants(self):
        if len(self.parameters["original_parameters"]) == 0:
            for force_key in self.ff_optimizable.keys():
                if force_key != 'NBException':
                    for param_name, values in self.ff_optimizable[force_key][0].dtype.names:
                        if param_name not in ['atom1', 'atom2', 'atom3', 'atom4', 'periodicity']:
                            scaling_constant = 1 / (np.mean(self.ff_optimizable[force_key][0][param_name][:self.n_atoms]))
                            scaling_constants = np.full(self.n_atoms, scaling_constant)
                            scaled_parameters = self.ff_optimizable[force_key][0][param_name][:self.n_atoms] * scaling_constants

                            self.parameters["original_parameters"].append(self.ff_optimizable[force_key][0][param_name][:self.n_atoms])
                            self.parameters["scaling_constants"].append(scaling_constants)
                            self.parameters["scaled_parameters"].append(scaled_parameters)

                            self.parameters["current_params"].append(self.ff_optimizable[force_key][0][param_name][:self.n_atoms])
                            self.parameters["current_scaling_constants"].append(scaling_constants)
                            self.parameters["current_scaled_parameters"].append(scaled_parameters)
        else:
            for force_key in self.ff_optimizable.keys():
                for param_type_no in range(len(self.ff_optimizable[force_key][0].dtype.names)):
                    current_scaling_constant = 1 / (np.mean(self.parameters["current_params"][:self.n_atoms * param_type_no]))
                    current_scaling_constants = np.full(self.n_atoms, current_scaling_constant)
                    current_scaled_params = self.parameters["current_params"][:self.n_atoms * param_type_no] * current_scaling_constants

                    self.parameters["current_scaling_constants"].append(current_scaling_constants)
                    self.parameters["current_scaled_parameters"].append(current_scaled_params)

    def calc_force_std_dev(self):
        #to calculate the standard deviation of forces
        delta_F = self.fmm - self.fqm
        force_std_dev = np.std(delta_F)
        return force_std_dev
    def wrap_objective_function(self):
        #to create the objective function for optimization
        
        for frame_nr, frame in enumerate(coords1=0):
            #set the coordinates for this frame
            self.molecular_system.set_coordinates(frame)  # You need to adjust this based on your code
            
            #unscale the parameters
            unscaled_params = self.unscale_parameters(self.parameters['current_scaled_parameters'])
            
            #set the unscaled parameters in the OpenMM system and recalculate properties
            self.recalculate_mm_properties(unscaled_params)  # might need to modify this part based on the necessary parameters
            
        if self.term_type == 'energy':
            obj_f = self.calculate_obj_func_energy()

        elif self.term_type == 'force':
            obj_f = self.calculate_obj_func_force()

        elif self.term_type == 'force & energy':
            obj_f_f = self.calculate_obj_func_force()
            obj_f_e = self.calculate_obj_func_energy()
            obj_f = obj_f_f + obj_f_e

        #add regularization term if needed
        if self.regularization:
            regularization_term = self.calculate_regularization_term()
            obj_f += regularization_term

        return obj_f

    def unscale_parameters(self, scaled_params):
        
        unscaled_params = []
        for scaled_param, scaling_constant in zip(scaled_params, self.parameters["current_scaling_constants"]):
            unscaled_param = scaled_param / scaling_constant
            unscaled_params.append(unscaled_param)
        return unscaled_params

    def parametrize(self, regularization_type='L2', scaling_factor=1.0, hyperbolic_beta=0.01):
        assert self.optimizer is not None, 'optimizer not set'
        assert self.term_type is not None, 'term_type not set'
        assert self.term_type in self.term_types, 'please choose a valid term_type'

        self.calc_scaling_constants()

        #implement regularization term if needed
        if self.regularization:
            regularization_term = self.calculate_regularization_term(regularization_type, scaling_factor, hyperbolic_beta)
        else:
            regularization_term = 0.0

        parameters = self.parameters['scaled_parameters']

        optimized_params, obj_f_value = self.optimizer.run_optimizer(self.wrap_objective_function, parameters)
        self.parameters['current_scaled_parameters'] = optimized_params

        obj_f_value += regularization_term
