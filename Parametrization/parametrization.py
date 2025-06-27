#### package import ####

import numpy as np
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
        Additionally, 'force_i' can be used to enable an individal variance computation for each conformation (some sort of wheighing).
        'force_c' computes the variance over all conformations and uses the same value for each conformation in the objective function. 
        This is the default setting also used in 'force' and 'force & energy'.
    optimizer : Parametrization.optimiizers.Optimizer object
        Contains optimizer type and specific settings
    constraints : str
        Applies manual 'man' or automatic 'auto' bounds directly or indirectly to the optimization
        sets self.bounds
    
    other (internal) parameters:
        self.parameters : 1d array
            contains optimized, scaled & vectorized parameters
        self.bounds : list of tuples
            upper and lower limit for parameters if the optimization is constrained.
    """
    def __init__(self, molecular_system=None, term_type=None, optimizer=None, constraints=None):

        assert molecular_system is not None, 'no molecular system selected'

        self.molecular_system = molecular_system #TODO: implement constraints checker
        self.emm = self.molecular_system.mm_energies['all']
        self.fmm = self.molecular_system.mm_net_forces
        self.eqm = self.molecular_system.qm_energies['all']
        self.fqm = self.molecular_system.qm_net_forces
        self.weights = self.molecular_system.weights
        self.n_atoms = self.molecular_system.n_atoms
        self.parameters = None

        self.optimizer = optimizer
        self.constraints = constraints

        if self.constraints == 'auto':

            if 'NonbondedForce' in self.molecular_system.reduced_indexed_ff_optimizable['all'].keys():

                assert len(list(self.molecular_system.reduced_indexed_ff_optimizable['all'].keys())) == 1, 'Automatic constraints support only NonbondedForce.'

                sigma_constraints = (0.03, 5.0)
                epsilon_constraints = (0.1, 5.0)

                bounds = []

                for parameter_type in self.molecular_system.reduced_indexed_ff_optimizable['all']['NonbondedForce'][0]:

                    if parameter_type[1][0] == 'P':

                        bounds.append((0.0, 6.0))
                        bounds.append(sigma_constraints)
                        bounds.append(epsilon_constraints)

                    elif parameter_type[1][0] == 'H':

                        bounds.append((0.0, 1.0))
                        bounds.append(sigma_constraints)
                        bounds.append(epsilon_constraints)

                    elif parameter_type[1][0] == 'C':

                        bounds.append((-4.0, 4.0))
                        bounds.append(sigma_constraints)
                        bounds.append(epsilon_constraints)

                    elif parameter_type[1][0] == 'O':

                        bounds.append((-2.0, 0.0))
                        bounds.append(sigma_constraints)
                        bounds.append(epsilon_constraints)

                    elif parameter_type[1][0] == 'N':

                        bounds.append((0.0, -3.0))
                        bounds.append(sigma_constraints)
                        bounds.append(epsilon_constraints)

                    else: 
                        raise ValueError('Automatic bounds for {} charge not implemented'.format(parameter_type[1][0]))

                self.bounds = bounds    

        if self.constraints == 'man':
            print('Please register a list of tuples with len(parameters) in Parametrization.bounds')

        self.term_types = ['energy', 'force', 'force & energy', 'force_i', 'force_c']
        self.term_type = term_type

        self.iterations = 0


    def calculate_classical_energies_forces(self):
        """
        calculates all classical energies and forces using OpenMM for the molecular_system

        sets :
            self.emm
        """

        for omm_system_name in self.molecular_system.openmm_systems.keys():
            if self.molecular_system.openmm_systems[omm_system_name] != None:
                self.molecular_system.generate_mm_energies_forces(omm_system_name)
                #print('mm_energies_forces generated')

        self.emm = self.molecular_system.mm_energies['all']

    def calculate_classical_net_forces(self):
        """
        calculates classical net forces (if more than one molecule/ molecule is solvated)

        sets :
            self.fmm
        """

        self.molecular_system.calculate_mm_net_forces()
        #print('mm_net_forces calculated')

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

        if self.optimizer.opt_method in ["scipy_local", "scipy_global", "bayesian", "cma", "pso"]:

            delta_E = self.emm - self.eqm
            enumerator = np.power((delta_E - np.mean(delta_E)), 2)
            denominator = np.var(self.eqm)
            frac = enumerator / denominator
            obj_f_e = np.sum(self.weights * frac)
            # n_conf        (ΔE - <ΔE>)²
            #   Σ  ω_{conf} ------------
            #  conf         Var(E^{QM})            

            """

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

        elif self.optimizer.opt_method == "pt_opt":

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

            """
        else:
            raise ValueError('ERROR: Optimizer of type {} not implemented'.format(self.optimizer.opt_method.lower()))

        return obj_f_e

    def evaluate_obj_func_force(self, method=None): #TODO: Use another metric?
        """
        calculates the value of the objective function using the format required by the selected optimizer
                 1            n_atoms n_conf              |ΔF|²
        O = ----------------     Σ     Σ     ω_{conf} ------------
            3n_atoms n_confs   atom   conf             Var(F^{QM})

        returns :
            value of the objective function in the format of the selected optimizer
        """

        assert method in [None, "const_variance", "indiv_variance"], "Force property term for method {} is not implemented.".format(method)

        for sys_type in self.molecular_system.openmm_systems.keys():

            if self.molecular_system.openmm_systems[sys_type] != None:

                self.molecular_system.openmm_systems[sys_type].set_parameters()
                #print('params set for '+str(sys_type))

        if isinstance(self.weights, np.ndarray) == False:

            self.molecular_system.generate_weights()
            self.weights = self.molecular_system.weights
            weights = np.atleast_2d(self.weights) #weights weigh conformations
            weights = np.atleast_3d(weights.T)

        else: 

            weights = np.atleast_2d(self.weights) #weights weigh conformations
            weights = np.atleast_3d(weights.T)

        self.calculate_classical_energies_forces()
        self.calculate_classical_net_forces()
        #print('net forces calculated')

        if self.optimizer.opt_method in ["scipy_local", "scipy_global", "bayesian", "cma","pso"]:

            delta_F = self.fmm - self.fqm
            enumerator = np.power(np.abs(delta_F), 2)

            if method is None:
                method = "const_variance"

            if method == "const_variance":

                denominator = np.var(np.linalg.norm(self.fqm, axis=2)) # one var over all confs
                frac_weighted = weights * (enumerator / denominator)
                #frac_weighted = weights * enumerator # testing
                obj_f_f = np.sum(frac_weighted) / (3 * self.n_atoms['all'])
                #     1            n_atoms n_conf              |ΔF|²
                # ----------------     Σ     Σ     ω_{conf} ------------
                # 3n_atoms n_confs   atom   conf             Var(F^{QM})

                #print('ka ching')

            elif method == "indiv_variance":

                norm = np.linalg.norm(self.fqm, axis=2)
                variance = np.var(norm, axis=1)
                denominator = np.swapaxes(np.atleast_3d(variance),0,1) # individual var per conf
                frac_weighted = weights * (enumerator / denominator)
                #frac_weighted = weights * enumerator # testing
                obj_f_f = np.sum(frac_weighted) / (3 * self.n_atoms['all'])
                #     1            n_atoms n_conf              |ΔF|²
                # ----------------     Σ     Σ     ω_{conf} ------------
                # 3n_atoms n_confs   atom   conf             Var(F^{QM})                

            """
        elif self.optimizer.opt_method == "tf_adam":

            tf_fmm = tf.Variable(self.fmm, dtype=float)
            tf_fqm = tf.Variable(self.fqm, dtype=float)
            tf_weights = tf.constant(weights, dtype=float)
            tf_n_atoms = tf.constant(self.n_atoms['all'], dtype=float)
            delta_F = tf.math.subtract(tf_fmm, tf_fqm)

            if method is None:
                method = "variance"

            if method == "variance":

                enumerator = tf.math.pow(tf.math.abs(delta_F), 2)
                denominator = tf.math.reduce_variance(tf.linalg.normalize(tf_fqm, axis=2)[-1])
                frac_weighted = tf.math.multiply(tf_weights, (tf.math.divide(enumerator, denominator)))
                obj_f_f = tf.math.divide(tf.math.reduce_sum(frac_weighted), tf.math.multiply(3, tf_n_atoms))
                #     1            n_atoms n_conf              |ΔF|²
                # ----------------     Σ     Σ     ω_{conf} ------------
                # 3n_atoms n_confs   atom   conf             Var(F^{QM})
            

        elif self.optimizer.opt_method == "pt_opt":
              
            pt_fmm = torch.tensor(self.fmm, dtype=torch.float64, requires_grad=True)
            pt_fqm = torch.tensor(self.fqm, dtype=torch.float64, requires_grad=True)
            pt_weights = torch.tensor(weights, dtype=torch.float64, requires_grad=False)
            pt_n_atoms = torch.tensor(self.n_atoms['all'], dtype=torch.int64, requires_grad=False)
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
            """

        else:
            raise ValueError('ERROR: Optimizer of type {} not implemented'.format(self.optimizer.opt_method.lower()))
        
        return obj_f_f
    

    def calc_force_std_dev(self):
        """
        calculates the standard deviation of forces

        """
        delta_F = self.fmm - self.fqm
        force_std_dev = np.std(delta_F)

        return force_std_dev
    
    
    def wrap_objective_function(self, parameters): 
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
        #print('params handed over')

        self.molecular_system.unscale_parameters()
        self.molecular_system.redistribute_vectorized_parameters()
        self.molecular_system.reshape_vectorized_parameters()
        self.molecular_system.expand_reduced_parameters()
            
        if self.term_type == 'energy':
            obj_f_value = self.evaluate_obj_func_energy()

        elif self.term_type == 'force':
            obj_f_value = self.evaluate_obj_func_force()

        elif self.term_type == 'force_c':
            #print('calculating_forces')
            obj_f_value = self.evaluate_obj_func_force(method="const_variance")

        elif self.term_type == 'force_i':
            obj_f_value = self.evaluate_obj_func_force(method="indiv_variance")

        elif self.term_type == 'force & energy':
            obj_f_f = self.evaluate_obj_func_force()
            obj_f_e = self.evaluate_obj_func_energy()
            obj_f_value = obj_f_f + obj_f_e


        self.molecular_system.reduce_ff_optimizable(self.molecular_system.slice_list)
        self.molecular_system.vectorize_reduced_parameters()
        self.molecular_system.merge_vectorized_parameters()
        self.molecular_system.scale_parameters()

        self.molecular_system.scaled_parameters = parameters
      
        self.iterations += 1

        return obj_f_value


    def parametrize(self, parameters):
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
        if self.constraints == 'man':
            assert len(self.bounds) == len(parameters), 'incomplete bounds set for parameters'
        
        if self.constraints != None:

            scaled_bounds = self.bounds*np.atleast_2d(self.molecular_system.scaling_factors).T
            for bound in scaled_bounds:
                bound.sort()

            if self.optimizer.opt_method.lower() == 'scipy_local':

                    self.optimizer.constraints = scaled_bounds   

            else:
                raise KeyError('Constraints not implemented for other optimizers')

        optimized_params, obj_f_value = self.optimizer.run_optimizer(self.wrap_objective_function, parameters)       
        self.parameters = optimized_params
