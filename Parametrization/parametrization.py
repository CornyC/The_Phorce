#### package import ####

import numpy as np
from Parametrization.optimizers import Optimizer
from System.system import *
import time

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

        self.molecular_system = molecular_system 
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

            bounds = []

            for force_group in self.molecular_system.reduced_indexed_ff_optimizable['all'].keys():

                """
                CAUTION! If you want to implement auto constraints for another force_group, make sure to put them in the alphabetically
                correct position, e.g. if you want to add HarmonicBondForce, put 'elif force_group == "HarmonicBondForce":' 
                after CustomNonbondedForce but before NonbondedForce. This ensures that the bounds list holds the corresponding 
                bounds in the correct position (alphabetic order)!
                """

                if force_group == 'CustomNonbondedForce':

                    r_sigma_constraints = (0.03, 1.0) # sigma in nm
                    epsilon_depth_constraints = (0.002, 20) # kJ/mol
                    nbfix_r_sigma_constr = (0.2, 1.0) # r_min in nm
                    nbfix_eps_depth_constr = (0.06, 15.5) # kJ/mol

                    for _array in self.molecular_system.reduced_indexed_ff_optimizable['all']['CustomNonbondedForce']:

                        for parameter_type in _array:
                                
                            if len(parameter_type) is 4:

                                if parameter_type[2][0] == 'P':

                                    bounds.append((r_sigma_constraints))
                                    bounds.append((epsilon_depth_constraints))

                                elif parameter_type[2][0] == 'H':

                                    bounds.append((r_sigma_constraints))
                                    bounds.append((epsilon_depth_constraints))

                                elif parameter_type[2][0] == 'C':

                                    bounds.append((r_sigma_constraints))
                                    bounds.append((epsilon_depth_constraints))

                                elif parameter_type[2][0] == 'O':

                                    bounds.append((r_sigma_constraints))
                                    bounds.append((epsilon_depth_constraints))

                                elif parameter_type[2][0] == 'N':

                                    bounds.append((r_sigma_constraints))
                                    bounds.append((epsilon_depth_constraints))
                                
                                else: 
                                    raise NotImplementedError('Automatic bounds for {} charge not implemented'.format(parameter_type[1][0]))

                            elif len(parameter_type) is 5:

                                bounds.append((nbfix_r_sigma_constr))
                                bounds.append((nbfix_eps_depth_constr))

                            else:
                                raise ValueError('Parameter type {} not supported'.format(parameter_type))

                elif force_group == 'NonbondedForce':

                    if 'CustomNonbondedForce' not in self.molecular_system.reduced_indexed_ff_optimizable['all'].keys():
                        # Nonbonded parameters are handled by the NonbondedForce force group
                        sigma_constraints = (0.03, 1.0) # nm
                        epsilon_constraints = (0.002, 20.0) # kJ/mol

                    for _array in self.molecular_system.reduced_indexed_ff_optimizable['all']['NonbondedForce']:

                        for parameter_type in _array:

                            if parameter_type[1][0] == 'P':

                                bounds.append((0.0, 6.0))

                                if len(parameter_type) == 5:
                                    # Nonbonded parameters are handled by the NonbondedForce force group
                                    bounds.append(sigma_constraints)
                                    bounds.append(epsilon_constraints)

                            elif parameter_type[1][0] == 'H':

                                bounds.append((0.0, 1.0))

                                if len(parameter_type) == 5:

                                    bounds.append(sigma_constraints)
                                    bounds.append(epsilon_constraints)

                            elif parameter_type[1][0] == 'C':

                                bounds.append((-4.0, 4.0))

                                if len(parameter_type) == 5:

                                    bounds.append(sigma_constraints)
                                    bounds.append(epsilon_constraints)

                            elif parameter_type[1][0] == 'O':

                                bounds.append((-2.0, 0.0))

                                if len(parameter_type) == 5:

                                    bounds.append(sigma_constraints)
                                    bounds.append(epsilon_constraints)

                            elif parameter_type[1][0] == 'N':

                                bounds.append((0.0, -3.0))

                                if len(parameter_type) == 5:

                                    bounds.append(sigma_constraints)
                                    bounds.append(epsilon_constraints)

                            else: 
                                raise NotImplementedError('Automatic bounds for {} charge not implemented'.format(parameter_type[1][0]))

                else:
                    raise NotImplementedError('Automatic constraints support only NonbondedForce and CustomNonbondedForce.')

                self.bounds = bounds    

        if self.constraints == 'man':
            print('Please register a list with len(parameters) filled with tuples (lower_bound, upper_bound) in Parametrization.bounds')

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

        self.molecular_system.unscale_parameters()
        self.molecular_system.redistribute_vectorized_parameters()
        self.molecular_system.reshape_vectorized_parameters()
        self.molecular_system.expand_reduced_parameters()
            
        if self.term_type == 'energy':
            obj_f_value = self.evaluate_obj_func_energy()

        elif self.term_type == 'force':
            obj_f_value = self.evaluate_obj_func_force()

        elif self.term_type == 'force_c':
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

        print('Iteration {}'.format(self.iterations), end = '\r')
        time.sleep(1)

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
