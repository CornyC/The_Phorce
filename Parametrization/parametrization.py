#### package import ####

import numpy as np
import tensorflow as tf
import torch
from Parametrization.optimizers import Optimizer

#### parametrizes your system ####

class Parametrization:
    """
    Parametrizes your molecular system.

    Parameters
    ----------
    molecular_system : Molecular_system object
        Contains properties of molecular system that is to be parametrized
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
    # TODO: how to set self.n_atoms smoothly???

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

        self.parameters = { 'original_parameters': [],
                            'scaling_constants': [],
                            'scaled_parameters': [],
                            'current_params': [],
                            'current_scaling_constants': [],
                            'current_scaled_parameters': []
        }

        self.term_types = ['energy', 'force', 'force&energy']
        self.term_type = term_type

        self.step = 0

    def recalculate_mm_properties(self):

        #TODO: set Params in openmm according to ff_optimizable
        for force_key in self.ff_optimizable.keys():
            self.molecular_system.openmm_sys.set_parameters(force_key)

        self.molecular_system.get_mm_energies_forces()

        #update instances
        self.emm = self.molecular_system.mm_energies
        self.fmm = self.molecular_system.mm_forces

    def calculate_obj_func_energy(self):
        """
         Calculates the "difference" between QM and MM energies

         Parameters
         ----------
         method : str
             mathematical method to calculate objective function. Default is None, covariance does not work.

         self-Params:
         self.weights : np.array, determines importance or influence of conformations
         self.optimizer.opt_method : how to handle optimization. SciPy-based methods and PSO use numpy,
                                     TensorFlow and PyTorch have their own math functions.
         self.fmm : np.array, classical energies on atoms of conformations
         self.fqm : np.array, quantum energies on atoms of conformations
         self.n_atoms : int, number of atoms of interest

         Returns "difference" between QM and MM energies as float
        """

        if self.optimizer.opt_method == ("scipy" or "pso"):

            # E^{MM} - E^{QM} = ΔE
            delta_E = self.emm - self.eqm
            # (ΔE - <ΔE>)²
            enumerator = np.power((delta_E - np.mean(delta_E)), 2)
            # Var(E^{QM})
            denominator = np.var(self.eqm)
            # (ΔE - <ΔE>)² / Var(E^{QM})
            frac = enumerator / denominator
            # n_conf        (ΔE - <ΔE>)²
            #   Σ  ω_{conf} ------------
            #  conf         Var(E^{QM})
            obj_f_e = np.sum(self.weights * frac)

        elif self.optimizer.opt_method == "tf_adam":

            tf_emm = tf.Variable(self.emm, dtype=float)
            tf_eqm = tf.Variable(self.eqm, dtype=float)
            tf_weights = tf.constant(self.weights, dtype=float)

            # E^{MM} - E^{QM} = ΔE
            delta_E = tf.math.subtract(tf_emm, tf_eqm)
            # (ΔE - <ΔE>)²
            enumerator = tf.math.pow((tf.math.subtract(delta_E, tf.math.reduce_mean(delta_E))), 2)
            # Var(E^{QM})
            denominator = tf.math.reduce_variance(tf_eqm)
            # (ΔE - <ΔE>)² / Var(E^{QM})
            frac = tf.math.divide(enumerator, denominator)
            # n_conf        (ΔE - <ΔE>)²
            #   Σ  ω_{conf} ------------
            #  conf         Var(E^{QM})
            obj_f_e = tf.math.reduce_sum(tf.math.multiply(tf_weights, frac))

        elif self.optimizer.opt_method == ("pt_adam" or "pt_lbfgs"):

            pt_emm = torch.tensor(self.emm, dtype=torch.float64, requires_grad=True)
            pt_eqm = torch.tensor(self.eqm, dtype=torch.float64, requires_grad=True)
            pt_weights = torch.tensor(self.weights, dtype=torch.float64, requires_grad=False)

            # E^{MM} - E^{QM} = ΔE
            delta_E = pt_emm - pt_eqm
            # (ΔE - <ΔE>)²
            enumerator = torch.pow((delta_E - torch.mean(delta_E)), 2)
            # Var(E^{QM})
            denominator = torch.var(pt_eqm)
            # (ΔE - <ΔE>)² / Var(E^{QM})
            frac = torch.div(enumerator, denominator)
            # n_conf        (ΔE - <ΔE>)²
            #   Σ  ω_{conf} ------------
            #  conf         Var(E^{QM})
            obj_f_e = torch.sum(torch.mul(pt_weights, frac))

        else:
            print('optimizer of type {} not implemented'.format(Optimizer.opt_method.lower()))

        return obj_f_e

    def calculate_obj_func_force(self, method=None):
        """
        Calculates the "difference" between QM and MM forces

        Parameters
        ----------
        method : str
            mathematical method to calculate objective function. Default is None, covariance does not work.

        self-Params:
        self.weights : np.array, determines importance or influence of conformations
        self.optimizer.opt_method : how to handle optimization. SciPy-based methods and PSO use numpy,
                                    TensorFlow and PyTorch have their own math functions.
        self.fmm : np.array, classical forces on atoms of conformations
        self.fqm : np.array, quantum forces on atoms of conformations
        self.n_atoms : int, number of atoms of interest

        Returns "difference" between QM and MM forces as float
        """

        assert method in [None, "variance", "covariance"], \
            "Force property term for method {} is not implemented.".format(method)

        weights = np.atleast_2d(self.weights)
        weights = np.atleast_3d(weights.T)

        if self.optimizer.opt_method == ("scipy" or "pso"):

            # F^{MM} - F^{QM} = ΔF
            delta_F = self.fmm - self.fqm

            if method is None:
                method = "variance"

            # version 1 using variance
            if method == "variance":

                # |ΔF|²
                enumerator = np.power(np.abs(delta_F), 2)
                # Var(F^{QM})
                denominator = np.var(np.linalg.norm(self.fqm, axis=2))
                #             |ΔF|²
                # ω_{conf} ------------
                #          Var(F^{QM})
                frac_weighted = weights * (enumerator / denominator)
                #     1      n_atoms,n_conf              |ΔF|²
                # --------         Σ        ω_{conf} ------------
                # 3n_atoms     atom,conf              Var(F^{QM})
                obj_f_f = np.sum(frac_weighted) / (3 * self.n_atoms)

            # version 2 using covariance

            elif method == "covariance":
                print('too fat to fly')

        elif self.optimizer.opt_method == "tf_adam":

            tf_fmm = tf.Variable(self.fmm, dtype=float)
            tf_fqm = tf.Variable(self.fqm, dtype=float)
            tf_weights = tf.constant(weights, dtype=float)
            tf_n_atoms = tf.constant(self.n_atoms, dtype=int)

            # F^{MM} - F^{QM} = ΔF
            delta_F = tf.math.subtract(tf_fmm, tf_fqm)

            if method is None:
                method = "variance"

            # version 1 using variance
            if method == "variance":

                # |ΔF|²
                enumerator = tf.math.pow(tf.math.abs(delta_F), 2)
                # Var(F^{QM})
                denominator = tf.math.reduce_variance(tf.linalg.normalize(tf_fqm, axis=2))
                #             |ΔF|²
                # ω_{conf} ------------
                #          Var(F^{QM})
                frac_weighted = tf.math.multiply(tf_weights, (tf.math.divide(enumerator, denominator)))
                #     1      n_atoms,n_conf              |ΔF|²
                # --------         Σ        ω_{conf} ------------
                # 3n_atoms     atom,conf              Var(F^{QM})
                obj_f_f = tf.math.divide(tf.math.reduce_sum(frac_weighted), tf.math.multiply(3, tf_n_atoms))

            # version 2 using covariance

            elif method == "covariance":
                print('too fat to fly')

        elif self.optimizer.opt_method == ("pt_adam" or "pt_lbfgs"):

            pt_fmm = torch.tensor(self.fmm, dtype=torch.float64, requires_grad=True)
            pt_fqm = torch.tensor(self.fqm, dtype=torch.float64, requires_grad=True)
            pt_weights = torch.tensor(weights, dtype=torch.float64, requires_grad=False)
            pt_n_atoms = torch.tensor(self.n_atoms, dtype=torch.int64, requires_grad=False)

            # F^{MM} - F^{QM} = ΔF
            delta_F = pt_fmm - pt_fqm

            if method is None:
                method = "variance"

            # version 1 using variance
            if method == "variance":

                # |ΔF|²
                enumerator = torch.pow(torch.abs(delta_F), 2)
                # Var(F^{QM})
                denominator = torch.var(torch.linalg.norm(pt_fqm, axis=2))
                #             |ΔF|²
                # ω_{conf} ------------
                #          Var(F^{QM})
                frac_weighted = torch.mul(pt_weights, torch.div(enumerator, denominator))
                #     1      n_atoms,n_conf              |ΔF|²
                # --------         Σ        ω_{conf} ------------
                # 3n_atoms     atom,conf              Var(F^{QM})
                obj_f_f = torch.sum(torch.div(frac_weighted, torch.mul(3, pt_n_atoms)))

            # version 2 using covariance

            elif method == "covariance":
                print('too fat to fly')

        else:
            print('optimizer of type {} not implemented'.format(Optimizer.opt_method.lower()))

        return obj_f_f

    def calc_scaling_constants(self):

        """
        This is supposed to bring the parameters to the same magnitude but idk
        """

        if len(self.parameters["original_parameters"]) is 0:
            for forcekey in self.ff_optimizable.keys():
                if forcekey is not 'NBException':
                    for param_name in self.ff_optimizable[forcekey][0].dtype.names:
                        if param_name not in ['atom1', 'atom2', 'atom3', 'atom4', 'periodicity']:
                            print(param_name)
                            scaling_constant = 1/(np.mean(self.ff_optimizable[forcekey][0][param_name][:self.n_atoms]))
                            scaling_constants = np.full(self.n_atoms, scaling_constant)
                            scaled_parameters = self.ff_optimizable[forcekey][0][param_name][:self.n_atoms] * scaling_constants
                            print(scaled_parameters)

                            self.parameters["original_parameters"].append(
                                self.ff_optimizable[forcekey][0][param_name][:self.n_atoms])
                            self.parameters["scaling_constants"].append(scaling_constants)
                            self.parameters["scaled_parameters"].append(scaled_parameters)

                            self.parameters["current_params"].append(self.ff_optimizable[forcekey][0][param_name][:self.n_atoms])
                            self.parameters["current_scaling_constants"].append(scaling_constants)
                            self.parameters["current_scaled_parameters"].append(scaled_parameters)

        else:
            for forcekey in self.ff_optimizable.keys():
                for param_type_no in range(len(self.ff_optimizable[forcekey][0].dtype.names)):
                    current_scaling_constant = 1/(np.mean(
                        self.parameters["current_params"][:(self.n_atoms * param_type_no)]))
                    current_scaling_constants = np.full(self.n_atoms, current_scaling_constant)
                    current_scaled_params = self.parameters["current_params"][:(self.n_atoms * param_type_no)] \
                                            * current_scaling_constants

                    self.parameters["current_scaling_constants"].append(current_scaling_constants)
                    self.parameters["current_scaled_parameters"].append(current_scaled_params)

    def calc_force_std_dev(self):

        # calc force difference
        delta_F = self.fmm - self.fqm

        force_std_dev = np.std(delta_F)

        return force_std_dev

    """
    def calc_regularization(self, regularization_type='L2', scaling_factor=1.0, hyperbolic_beta=0.01):

        
        #Penalty term that avoids overfitting. Scaling factor sets strength of regularization, use hyperbolic for charge
        #fitting. Uses mean as width of prior distribution of parameters.
        

        assert len(self.parameters["original_parameters"]) is not 0, 'No parameters stored in Parametrization. ' \
                                                                     'Run calc_scaling_constants method first.'
        assert type(regularization_type) == str, "No valid regularization method. Use L1, L2, or hyperbolic"
        assert regularization_type in ["L1", "L2", "hyperbolic"], "Regularization type {} is not implemented." \
            .format(regularization_type)

        no_of_parameter_types = self.parameters["original_parameters"] / self.n_atoms
        self.parameters['regularization'] = []

        if regularization_type is not "hyperbolic":

            change = (self.parameters['current_scaled_parameters'] - self.parameters['scaled_parameters']) \
                     / self.parameters['current_scaled_parameters']

            if regularization_type == "L1":
                # Lasso Regression
                regularization = np.abs(change)

            elif regularization_type == "L2":
                # Ridge Regression
                regularization = np.power(change, 2)

            for parameter_type_no in range(no_of_parameter_types):
                l_scaled_value = scaling_factor * np.sum(regularization[:(self.n_atoms * parameter_type_no)])
                l_scaled = np.full(self.n_atoms, l_scaled_value)
                self.parameters['regularization'].append(l_scaled)


        else:
            change = np.sqrt(np.power(self.parameters['current_scaled_parameters'], 2) + np.power(hyperbolic_beta, 2)) \
                     - hyperbolic_beta
            for parameter_type_no in range(no_of_parameter_types):
                regularization = np.sum(change[:(self.n_atoms * parameter_type_no)])
                hyp_scaled = scaling_factor * regularization
                self.parameters['regularization'].append(hyp_scaled)

        self.parameters['reg_cur_sc_parm'] = self.parameters['current_parameters_scaled'] \
                                             + self.parameters['regularization']

    """

    def wrap_objective_function(self):
        """
        This is the actual "function" that is passed to the optimizer.
        """
        # has to loop over configs, set coords & unscaled params in omm, calc mm energies/forces (repeatedly),
        # read qm energies/forces (once for each config)
        # calc obj fun: params as scaled array in, force diff out

        #TODO: unscale params???

        #TODO: loop over confs (should be function on its own)
        for frame_nr, frame in enumerate(coords1):

            # feed frame coords and new params into omm sys
            #self.molecular_system.openmm_sys # coords in
            self.recalculate_mm_properties() # params in, energies/forces out

        if self.term_type == 'energy':
            obj_f = self.calculate_obj_func_energy()

        elif self.term_type == 'force':
            obj_f = self.calculate_obj_func_force()

        elif self.term_type == 'force & energy':
            obj_f_f = self.calculate_obj_func_force()
            obj_f_e = self.calculate_obj_func_energy()
            obj_f = obj_f_f + obj_f_e

        # regularization term?
        """
        if regularization == True:
            obj_f += regularization_term
        """

        return obj_f # returns value

    def parametrize(self):

        assert self.optimizer is not None, 'optimizer not set'
        assert self.term_type is not None, 'term_type not set'
        assert self.term_type in self.term_types, 'pls choose a valid term_type'

        self.calc_scaling_constants()

        #TODO: this down below does not work, add 'term' to obj_f instead!
        """ 
        if self.regularization is True:
            self.calc_regularization()
            parameters = self.parameters['reg_cur_sc_parm']

        else:
            parameters = self.parameters['current_scaled_parameters']
            
        """
        parameters = self.parameters['scaled_parameters']

        optimized_params, obj_f_value = self.optimizer.run_optimizer(self.wrap_objective_function, parameters)
        self.parameters['current_scaled_parameters'] = optimized_params



