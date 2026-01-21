#### package imports ####
import numpy as np
from pathlib import Path
from System.paths import *

class Optimizer:
    """
    Creates the optimizer for the parametrization problem.

    Parameters
    ----------
    opt_method : str
        Name of the optimizer to be created. Available optimizers are "scipy_local" (as in all local scipy optimizers),
        "scipy_global" (as in all global scipy optimizers), "bayesian" (ML-informed from the boss package), 
        "cma" (CMA-ES (Covariance Matrix Adaptation Evolution Strategy)), and "pso" (Particle swarm optimization).
    opt_settings : dict
        dictionary containing the optimizer-specific settings. See the respective docs for info.
    max_iterations : int
        maximum optimization steps the optimizer should take, default = 10000
    tolerance : float
        change value below which the optimizer is supposed to be converged, default = 1e-8
    constraints : list of tuples
        gets passed on from System.Molecular_system.constraints to the optimizer if it supports either bounds or a constraint function.
        Otherwise, a penalty term is added to the objective function.
        Default = None
    enforce_constraints : bool
        Default = False, applies penalty term to objective function for scipy Nelder-Mead
    f : Parametrization.parametrization.Objective_function object
        Objective function
    parameters : numpy array 
        Parameters that are going to be optimized (extracted from ff_optimizable using parametrization.Parametrization)
    scipy_optimization : scipy.optimize object
        specific optimizer, sometimes handy to check status & output
    cma_optimization : pycma object
        specific optimizer, sometimes handy to check status & output
    """

    optimizers = ["scipy_local",
                  "scipy_global",
                  "bayesian",
                  "cma",
                  "pso"]

    def __init__(self, opt_method, opt_settings, max_iterations=10000, tolerance=1e-8, constraints=None):

        assert opt_method.lower() in self.optimizers, 'optimizer of type {} not implemented'.format(opt_method.lower())

        self.opt_method = opt_method
        self.opt_settings = opt_settings
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.constraints = constraints
        self.enforce_constraints = False
        self.iterations = 0

        # other attributes
        self.f = None
        self.parameters = None
        # optimizer objetcs
        self.scipy_optimization = None
        self.cma_optimization = None

    def constraints_func(self, parameters):

        """
        Penalty function for constrained optimization which is handed to the optimizer directly if supported or added to the objective function.
        Checks if parameters are within bounds and if not adds penalty.
        """

        assert self.constraints is not None, 'Cannot construct constraints function from empty bounds'
        assert len(self.constraints) == len(parameters), 'number of bounds and number of parameters do not match'

        within_bounds = (parameters >= self.constraints[:,0]) & (parameters <= self.constraints[:,1])
        penalty = np.sum(~within_bounds) * (self.iterations + 1)

        self.iterations += 1

        return penalty
    
    def penalize_objective_function(self, parameters):

        penalized_obj_func_value = self.f(parameters) + self.constraints_func(parameters)

        return penalized_obj_func_value

    def optimize_with_scipy_local(self):

        """
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        Nelder-Mead is recommended, Powell or COBYLA could work, too. Set the specific optimizer thru 
        Opti = Optimizer('scipy_local', {'method': 'Nelder-Mead'}) during initialization of the Optimizer object.

        Attributes
        ----------
        parameters : np.array
            parameters that go into the function
        opt_settings : dict
            optimizer-speific settings
        constraints : np.array
            idk yet
        tolerance : float
            optimizer tolerance
        f : parametrization.Parametrization.wrap_objective_function
            Objective function wrapper

        Returns
        -------
        optimized_params, value 
            np.array containing the optimized force field params, objective function value
        """

        from scipy.optimize import minimize as scmin

        self.opt_settings.update({'tol': self.tolerance})
        self.opt_settings.update({'options' : {'maxiter': self.max_iterations}})

        if self.constraints is not None:

            if self.opt_settings['method'] not in ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC','SLSQP', 'trust-constr', 'SLSQP', 'trust-constr', 'COBYLA']:
                # apply penalty term to objective function
                optimization = scmin(lambda x : self.penalize_objective_function(x), x0=self.parameters, **self.opt_settings)

            elif self.opt_settings['method'] in ['Powell', 'L-BFGS-B', 'TNC','SLSQP', 'trust-constr', 'SLSQP', 'trust-constr', 'COBYLA']:
                # apply bounds to optimizer
                if self.opt_settings['method'] in ['Powell', 'L-BFGS-B', 'TNC','SLSQP', 'trust-constr']:
                    self.opt_settings.update({'bounds': self.constraints})

                    if self.opt_settings['method'] in ['SLSQP', 'trust-constr']:
                        # apply constraint func to optimizer, too
                        constraints = [{'type': 'eq', 'fun': lambda x: self.constraints_func(x)}]
                        self.opt_settings.update({'constraints': constraints})

                elif self.opt_settings['method'] == 'COBYLA':    
                    # apply only constraint func to optimzer
                    constraints = [{'type': 'eq', 'fun': lambda x: self.constraints_func(x)}]
                    self.opt_settings.update({'constraints': constraints})
                # call normal opt procedure
                optimization = scmin(lambda x : self.f(x), x0=self.parameters, **self.opt_settings)

            elif self.opt_settings['method'] == 'Nelder-Mead':
                # apply bounds to optimizer (Nelder-Mead needs bounds outside the options dict)
                self.opt_settings.update({'bounds': self.constraints})

                if self.enforce_constraints == True:
                    # sometimes bounds are violated, enforce them via penalty term @ objective func
                    optimization = scmin(lambda x : self.penalize_objective_function(x), x0=self.parameters, **self.opt_settings)

                else:
                    # call normal opt procedure
                    optimization = scmin(lambda x : self.f(x), x0=self.parameters, **self.opt_settings)

        else:
            # call normal opt procedure
            optimization = scmin(lambda x : self.f(x), x0=self.parameters, **self.opt_settings)

        optimized_params = optimization.x
        value = optimization.fun

        self.scipy_optimization = optimization

        return optimized_params, value
    
    def optimize_with_scipy_global(self): #TODO: test

        """
        see https://docs.scipy.org/doc/scipy/reference/optimize.html

        Attributes
        -----------

        parameters : np.array
            parameters that go into the function
        opt_settings : dict
            nested dict of optimizer-speific settings, see above url
        tolerance : float
            optimizer tolerance
        f : parametrization.Parametrization.wrap_objective_function
            Objective function wrapper

        """
        import scipy.optimize as s_opt

        assert self.opt_settings['method'] in \
            ["basinhopping", "brute", "differential_evolution", "shgo", "dual_annealing", "direct"],\
            'scipy global optimizer {} does not exist'.format(self.opt_settings['method'])
        
        if self.opt_settings['method'] == "basinhopping":

            self.opt_settings.update({'options' : {'niter': self.max_iterations}})
            
            if self.opt_settings['local_method'] != None:
                minimizer_kwargs['method'] = self.opt_settings['local_method']

            elif minimizer_kwargs == None:
                minimizer_kwargs = {'method': 'Nelder-Mead'} # local optimizer

            optimization = s_opt.basinhopping(lambda x : self.f(x), self.parameters, minimizer_kwargs=minimizer_kwargs, niter=self.max_iterations)

            optimized_params = optimization.x 
            value = optimization.fun

        elif self.opt_settings['method'] == "brute":

            optimization = s_opt.brute(lambda x : self.f(x), (self.parameters*1e-3, self.parameters*1e1), args=None, Ns=len(self.parameters*2), full_output=True, workers=-1)
            optimized_params = optimization[0]
            value = optimization[1]

        elif self.opt_settings['method'] == "differential_evolution":

            self.opt_settings.update({'options' : {'maxiter': self.max_iterations}})
            self.opt_settings.update({'options' : {'tol': self.tolerance}})
            self.opt_settings.update({'options' : {'workers': -1}})

            optimization = s_opt.differential_evolution(lambda x : self.f(x), (self.parameters*1e-3, self.parameters*1e1), **self.opt_settings['options'], x0=self.parameters )

            optimized_params = optimization.x
            value = optimization.fun

        elif self.opt_settings['method'] == "shgo":

            self.opt_settings.update({'options': {'maxiter': self.max_iterations}})
            self.opt_settings.update({'options': {'f_tol': self.tolerance}})

            if self.opt_settings['local_method'] != None:
                minimizer_kwargs['method'] = self.opt_settings['local_method']

            elif minimizer_kwargs == None:
                minimizer_kwargs = {'method': 'Nelder-Mead'} # local optimizer

            optimization = s_opt.shgo(lambda x : self.f(x), (self.parameters*1e-3, self.parameters*1e1), args=None, minimizer_kwargs=minimizer_kwargs, **self.opt_settings, workers=-1)

            optimized_params = optimization.x
            value = optimization.fun

        elif self.opt_settings['method'] == "dual_annealing":

            if self.opt_settings['local_method'] != None:
                minimizer_kwargs['method'] = self.opt_settings['local_method']

            elif minimizer_kwargs == None:
                minimizer_kwargs = {'method': 'Nelder-Mead'} # local optimizer

            optimization = s_opt.dual_annealing(lambda x : self.f(x), (self.parameters*1e-3, self.parameters*1e1), args=None, maxiter=self.max_iterations, minimizer_kwargs=minimizer_kwargs, x0=self.parameters)

            optimized_params = optimization.x
            value = optimization.fun

        elif self.opt_settings['method'] == "direct":

            optimization = s_opt.direct(lambda x : self.f(x), (self.parameters*1e-3, self.parameters*1e1), maxiter=self.max_iterations)

            optimized_params = optimization.x
            value = optimization.fun          

        self.scipy_optimization = optimization            

        return optimized_params, value


    def optimize_with_cma(self):

        """
        see https://cma-es.github.io/apidocs-pycma/
        CMA-ES is a stochastic optimizer for robust non-linear non-convex derivative- and function-value-free numerical optimization.
        It is terribly slow but might offer a last resort solution.

        Attributes
        -----------
        parameters : np.array
            parameters that go into the function
        opt_settings : dict
            optimizer-speific settings, if you want to use fancy settings, add them to the 'pycma settings' routine down below.
        """

        import cma

        sdev_of_params = np.std(self.parameters)

        # begin pycma settings
        self.opt_settings.update({'tolfun': self.tolerance})
        #self.opt_settings.update({'maxiter': self.max_iterations}) # the definition in the pycma docs is weird

        for setting in self.opt_settings.keys():

            if setting == 'tolfun': 

                cma.CMAOptions().set('tolfun', self.tolerance)

            elif setting == 'maxiter':

                cma.CMAOptions().set('maxiter', self.max_iterations)
        # end pycma settings
        
        assert 'method' in list(self.opt_settings.keys()), 'No pycma method set.'
        assert self.opt_settings['method'] in ['EvolutionStrategy', 'fmin2'], 'pycma method {} not implemented.'.format(self.opt_settings['method'])

        # the folloing is adapted from the pycma doc, not sure if it makes sense   
        if self.opt_settings['method'] == 'EvolutionStrategy':

            evo_strat = cma.CMAEvolutionStrategy(self.parameters, sdev_of_params)
            evo_strat.optimize(lambda x : self.f(x))

        elif self.opt_settings['method'] == 'fmin2':

            paramsout, evo_strat = cma.fmin2(lambda x: self.f(x), self.parameters, sdev_of_params) # paramsout yields the same solutions as method = EvolutionStrategy
            evo_strat = cma.CMAEvolutionStrategy(self.parameters, sdev_of_params).optimize(lambda x: self.f(x))
            # this seems to run the optimizer twice and finds different solutions

        optimized_params = evo_strat.result.xbest
        value = evo_strat.result.fbest

        self.cma_optimization = evo_strat

        return optimized_params, value


    def optimize_with_boss(self): #TODO: test
        """
        Bayesian optimizer. See https://cest-group.gitlab.io/boss/index.html.
        Bounds are created automatically based on self.parameters.  

        Attributes
        ----------
        parameters : np.array
            parameters that go into the function
        opt_settings : dict
            optimizer-speific settings, {'options': {'bo_output': outputpath}} needs to be set. 
            Most BOMain keywords can be passed through the options dictionary. If yours is missing, implement it 
            down below like the others.
        """

        from boss.bo.bo_main import BOMain
        from boss.pp.pp_main import PPMain

        if "options" in list(self.opt_settings.keys()):

            if "bo_outfile" in list(self.opt_settings['options'].keys()):

                bo_outfile = self.opt_settings['options']['bo_outfile']
                assert Path(bo_outfile).is_dir() == True, 'boss output filepath {} does not exist'.format(bo_outfile)
                
                if bo_outfile[-1] != '/':
                    bo_outfile += '/'

            else: 
                raise KeyError("please specify 'bo_output': outputpath in Optimizer.opt_settings['options'].")
            
            if 'kernel' in list(self.opt_settings['options'].keys()):
                bo_kernel = self.opt_settings['options']['kernel']
            else:
                bo_kernel = 'rbf'

            if 'initpts' in list(self.opt_settings['options'].keys()):
                bo_initpts = self.opt_settings['options']['initpts']
            else:
                bo_initpts = 20

            if 'noise' in list(self.opt_settings['options'].keys()):
                bo_noise = self.opt_settings['options']['noise']
            else:
                bo_noise = 1e-4

            if 'acqfn_name' in list(self.opt_settings['options'].keys()):
                bo_acqfn = self.opt_settings['options']['acqfn_name']
            else:
                bo_acqfn = 'exploit'

            if 'acqtol' in list(self.opt_settings['options'].keys()):
                bo_acqtol = self.opt_settings['options']['acqtol'] 
            else:
                bo_acqtol = 1e-3

        else: 
            raise KeyError("please specify an output path for boss in Optimizer.opt_settings['options'] = {'bo_outfile': outputpath}.")

        bounds_low = np.full(len(self.parameters), 1e-3)
        bounds_high = np.full(len(self.parameters), 1e0)
        bounds_comb = np.vstack((bounds_low, bounds_high)).T 

        optimization = BOMain(
            lambda x : self.f(x),
            bounds = bounds_comb,
            noise = bo_noise,
            initpts = bo_initpts,
            iterpts = self.max_iterations,
            kernel = bo_kernel,
            acqfn_name = bo_acqfn,
            acqtol = bo_acqtol,
            outfile = bo_outfile+'boss.out',
            rstfile = bo_outfile+'boss.rst',
            )

        optimization_result = optimization.run(self.parameters)    
        self.bayesian_optimization = optimization_result #how does this look?

        params_acquisition = optimization_result.get_next_acq(-1) #?
        value = optimization_result.select("Y", [-1]) #does this work?

        postprocessing = PPMain(optimization_result, pp_acq_funcs=True, pp_models=True) 
        self.bo_pp = postprocessing #do i need this?

        return params_acquisition.flatten(), value #idk


    def optimize_with_pso(self): 
        """
        Particle swarm optimizer. Beware this feature is experimental. Choose hyperparameters carefully. Set them by
        c1, c2, w, n_particles = Optimizer.opt_settings.values() after having initialized the Optimizer object and before running 
        the parametrization loop.

        Parameters
        ----------
        c1, c2, w : float
            Hyperparameters used in particle swarm optimization, defaults: c1=0.1, c2=0.1, w=0.4
        n_particles : int
            number of swarm particles to be generated, default = 50

        Attributes
        ----------
        parameters : np.array
            parameters that go into the function
        f : 
            input function (wrapped objective function)
        max_iterations : int
            limits the number of iterations
        tolerance 
            below which change the optimization is defined as converged

        Returns
        -------
        optimized_params : np.array 
            contains the optimized force field params
        value : float
            value of the input function self.f
        """
        assert ['c1', 'c2', 'w', 'n_particles'] == list(self.opt_settings.keys()), 'pso optimizer needs c1, c2, w, n_particles as opt_settings'

        c1, c2, w, n_particles = self.opt_settings.values()

        print('PSO hyperparameters: c1 = {}, c2 = {}, w = {}'.format(c1, c2, w))
        print('n_particles set to {}'.format(n_particles))

        import random

        # create particles

        np.random.seed(100)

        particle_positions0 = np.zeros((n_particles, len(self.parameters))) #init array
        func_value_particles0 = np.zeros((n_particles)) #init array

        for particle in range(n_particles): 
            # fill array
            distorted_positions = np.array(random.choices(self.parameters, k=len(self.parameters)))
            particle_positions0[particle] = distorted_positions

        particle_velocities0 = np.random.randn(self.parameters.size) * 0.1
        particle_positions1 = particle_positions0 + particle_velocities0

        # evaluate initial data
        func = lambda parameters: self.f(parameters) 
   
        for particle in range(n_particles):
            # fill array
            func_value_particles0[particle] = func(particle_positions0[particle])

        best_positions0 = particle_positions0[np.argmin(func_value_particles0)]

        # initial data
        particle_positions_prev = particle_positions0
        particle_positions_current = particle_positions1
        particle_velocities_prev = particle_velocities0
        func_value_particles_prev = func_value_particles0
        best_position_prev = best_positions0

        change = np.array(1.)
        i = 0

        while (i < self.max_iterations and change > self.tolerance and change > 0):
            print(i)
            # random numbers r1, r2 for velocity generation
            r1, r2 = np.random.rand(2)

            # generate new velocities
            particle_velocities_current = w * particle_velocities_prev + \
                                  c1 * r1 * (particle_positions_current - particle_positions_prev) + \
                                  c2 * r2 * (-particle_positions_prev + best_position_prev)

            # move particles
            particle_positions_new = particle_positions_current + particle_velocities_current

            # reassign vars
            particle_positions_prev = particle_positions_current
            particle_positions_current = particle_positions_new
            func_value_particles_current = np.zeros((n_particles))

            # evaluate old and new positions
            for particle in range(n_particles):
                func_value_particles_prev[particle] = func(particle_positions_prev[particle])
                func_value_particles_current[particle] = func(particle_positions_current[particle])

            best_position_prev = particle_positions_prev[np.argmin(func_value_particles_prev)]
            min_func_prev = np.min(func_value_particles_prev)
            
            # update prev params by better ones from current
            particle_positions_prev[(func_value_particles_prev >= func_value_particles_current)] = particle_positions_current[(func_value_particles_prev >= func_value_particles_current)]
            # find smallest values for both the new func value and the old func value
            func_opt = np.min(np.array([func_value_particles_prev, func_value_particles_current]), axis=0)
            # slice positions by smallest values for func -> best positions
            best_position_current = particle_positions_prev[np.argmin(func_opt)]
            # calc smallest func value
            min_func = np.min(func_opt)
            
            # assure progress
            change = min_func_prev - min_func

            i += 1
        
        #optimized_params = particle_positions
        #value = func
        value = min_func
        optimized_params = best_position_current

        print("PSO found the best solution at obj_func({})={}".format(optimized_params, value))

        return optimized_params, value

    def run_optimizer(self, f, parameters, constraints=None):
        """
        Parameters
        ----------
        f : Parametrization.parametrization.Parametrization object
            Objective function
        parameters : numpy array 
            Parameters that are going to be optimized (extracted from ff_optimizable)
        constraints :
            Optimization constraints, default = None

        Returns
        -------
        optimized_params : np.array 
            contains the optimized force field params
        value : float 
            value of self.f
        """

        self.f = f
        self.parameters = parameters
        self.contsraints = constraints

        print("###################################################################################")
        print(
            "#                         STARTING " + self.opt_method.upper() + " OPTIMIZER                            ")
        print("###################################################################################")

        if self.opt_method == "scipy_local":

            optimized_params, value = self.optimize_with_scipy_local()

        elif self.opt_method == "scipy_global":

            optimized_params, value = self.optimize_with_scipy_global()

        elif self.opt_method == 'cma':

            optimized_params, value = self.optimize_with_cma()

        elif self.opt_method == 'bayesian':

            optimized_params, value = self.optimize_with_boss()

        elif self.opt_method == 'pso':

            optimized_params, value = self.optimize_with_pso()

        print("###################################################################################")
        print(
            "#                         " + self.opt_method.upper() + " OPTIMIZER FINISHED                            ")
        print("###################################################################################")

        print('optimized parameters:')
        print(optimized_params)
        print('objective function value:')
        print(value)

        return optimized_params, value






