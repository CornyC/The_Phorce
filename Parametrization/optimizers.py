#### package imports ####
import numpy as np
import copy


class Optimizer:
    """
    Creates the optimizer for the parametrization problem.

    Parameters
    ----------
    opt_method : str
        Name of the optimizer to be created. Available optimizers are "scipy" (as in all scipy optimizers),
        "tf_adam" (Tensorflow's Adam), "pt_adam" (PyTorch's Adam), "pt_lbfgs" (PyTorch's LBFGS),
        and "pso" (Particle swarm optimization).
    max_iterations : int
        maximum optimization steps the optimizer should take, default = 10000
    tolerance : float
        change value below which the optimizer is supposed to be converged, default = 1e-8
    opt_settings : list
        List containing the optimizer-specific settings. See the respective docs for info.
    f : Parametrization.parametrization.Objective_function object
        Objective function
    parameters : numpy array / tensorflow Variable / torch tensor
        Parameters that are going to be optimized (extracted from ff_optimizable using parametrization.Parametrization)
    constraints :
        Optimization constraints, default = None
    """

    optimizers = ["scipy",
                  "tf_adam",
                  "pt_adam",
                  "pt_lbfgs",
                  "pso"]

    def __init__(self, opt_method, opt_settings, max_iterations=10000, tolerance=1e-8, ):

        assert opt_method.lower() in self.optimizers, 'optimizer of type {} not implemented'.format(opt_method.lower())

        self.opt_method = opt_method
        self.opt_settings = opt_settings
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # other attributes
        self.f = None
        self.parameters = None
        self.constraints = None

    def optimize_with_scipy(self):

        """
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        BFGS is recommended, L-BFGS-B may be faster but more unreliable

        (internal) parameters :

        self.parameters : np.array
            parameters that go into the function
        self.opt_settings : list
            list of arguments that is extracted and passed to the minimizer
        self.constraints : np.array
            idk yet
        self.tolerance : float
            optimizer tolerance
        self.f : parametrization.Parametrization.wrap_objective_function
            Objective function wrapper

        returns:
            optimized_params : np.array containing the optimized force field params
        """

        from scipy.optimize import minimize as scmin

        self.opt_settings = {'tol': self.tolerance}
        self.opt_settings.update({'options' : { 'maxiter': self.max_iterations}})

        if self.constraints is not None:
            self.opt_settings.update({'constraints': self.constraints})

        optimization = scmin(lambda x : self.f(x), x0=self.parameters, **self.opt_settings)
        optimized_params = optimization.x
        value = optimization.fun

        return optimized_params, value

    def optimize_with_tf_adam(self): #TODO: fix self argument problem in self.f

        """
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam

        Parameters
        ----------
        self.parameters : parameters that go into the function
        self.opt_settings
        self.constraints
        self.tolerance
        self.f : input function (wrapped objective function)
        self.max_iterations

        returns:
            optimized_params : np.array containing the optimized force field params
            value : float value of the input function self.f
        """

        import tensorflow as tf

        tf_params = tf.Variable(self.parameters)
        i = 0
        opt = tf.keras.optimizers.Adam(*self.opt_settings)
        steps = []
        change = tf.Variable(np.array(1.))

        while (i < self.max_iterations and change > tf.constant(self.tolerance) and change > 0):
            f_0 = self.f(tf_params)

            with tf.GradientTape() as tp:
                cost_fn = self.f(tf_params)
            gradients = tp.gradient(cost_fn, [tf_params])
            opt.apply_gradients(zip(gradients, [tf_params]))

            tf_params_new = tf_params

            f_1 = self.f(tf_params_new)

            change = f_0 - f_1

            steps.append(np.array([tf_params]))

            i += 1

        optimized_params = [tf_params_new]
        value = f_1

        return optimized_params, value

    def optimize_with_pt_adam(self): #TODO: fix self argument problem in self.f

        """
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

        Parameters
        ----------
        self.parameters : parameters that go into the function
        self.opt_settings
        self.constraints
        self.tolerance
        self.f : input function (wrapped objective function)
        self.max_iterations

        returns:
            optimized_params : np.array containing the optimized force field params
            value : float value of the input function self.f
        """

        import torch

        pt_params = torch.tensor(self.parameters, dtype=torch.float64, requires_grad=True)
        i = 0
        opt = torch.optim.Adam(pt_params, self.opt_settings)
        steps = []
        change = torch.tensor(np.array(1.))

        while (i < self.max_iterations and change > self.tolerance and change > 0):

            for param in pt_params:
                opt.zero_grad()  # clear gradients to free mem
                func0 = self.f(param)
                func0.backward()  # computes gradients
                opt.step()

            pt_params_new = pt_params

            for param in pt_params_new:
                opt.zero_grad()  # clear gradients to free mem
                func1 = self.f(param)
                func1.backward()  # computes gradients
                opt.step()

            change = func0 - func1

            pt_res = [pt_params_new.detach().numpy()]
            steps.append([copy.deepcopy(pt_res)])

            i += 1

        optimized_params = pt_res
        value = func1.detach().numpy()

        return optimized_params, value

    def optimize_with_pt_lbfgs(self): #TODO: fix self argument problem in self.f

        """
        https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS

        Parameters
        ----------
        self.parameters : parameters that go into the function
        self.opt_settings
        self.constraints
        self.tolerance
        self.f : input function (wrapped objective function)
        self.max_iterations

        returns:
            optimized_params : np.array containing the optimized force field params
            value : float value of the input function self.f
        """

        import torch

        pt_params = torch.tensor(self.parameters, dtype=torch.float64, requires_grad=True)
        i = 0
        opt = torch.optim.LBFGS(pt_params, self.opt_settings)
        steps = []
        change = torch.tensor(np.array(1.))

        while (i < self.max_iterations and change > self.tolerance and change > 0):

            for param in pt_params:
                def closure():
                    opt.zero_grad()
                    func0 = self.f(param)
                    func0.backward()
                    return func0

                opt.step(closure)
                func0 = self.f(param)

            pt_params_new = pt_params

            for param in pt_params_new:
                def closure():
                    opt.zero_grad()  # clear gradients to free mem
                    func1 = self.f(param)
                    func1.backward()  # computes gradients
                    return func1

                opt.step(closure)
                func1 = self.f(param)

            change = func0 - func1

            pt_res = [pt_params_new.detach().numpy()]
            steps.append([copy.deepcopy(pt_res)])

            i += 1

        optimized_params = pt_res
        value = func1.detach().numpy()

        return optimized_params, value

    def optimize_with_pso(self, c1=0.1, c2=0.1, w=0.4, n_particles=100): #TODO: fix self argument problem in self.f
        """
        Parameters
        ----------
        c1, c2, w : float
            Hyperparameters used in particle swarm optimization
        n_particles : int
            number of swarm particles to be generated

        self.parameters : parameters that go into the function
        self.f : input function (wrapped objective function)
        self.max_iterations
        self.tolerance

        returns:
            optimized_params : np.array containing the optimized force field params
            value : float value of the input function self.f
        """

        import random

        # create particles
        np.random.seed(100)

        particle_positions0 = random.choices(self.parameters, k=n_particles)
        particle_velocities0 = np.random.randn(self.parameters.size, n_particles) * 0.1
        particle_positions1 = particle_positions0 + particle_velocities0

        # evaluate initial data
        func0 = self.f(particle_positions0)
        best_positions0 = np.min(func0)

        # initial data
        particle_positions_prev = particle_positions0
        particle_positions = particle_positions1
        particle_velocities_prev = particle_velocities0
        best_position_prev = best_positions0
        change = np.array(1.)
        i = 0

        while (i < self.max_iterations and change > self.tolerance and change > 0):
            # random numbers r1, r2 for velocity generation
            r1, r2 = np.random.rand(2)

            # generate new velocities
            particle_velocities = w * particle_velocities_prev + \
                                  c1 * r1 * (particle_positions - particle_positions_prev) + \
                                  c2 * r2 * (-particle_positions_prev + best_position_prev)

            # move particles
            particle_positions_new = particle_positions + particle_velocities

            # reassign vars
            particle_positions_prev = particle_positions
            particle_positions = particle_positions_new

            # evaluate old and new positions
            func_prev = self.f(particle_positions_prev)
            best_position_prev = particle_positions_prev[:, np.argmin(func_prev)]
            min_func_prev = np.min(func_prev)

            func = self.f(particle_positions)
            best_position = particle_positions[:, np.argmin(func)]
            min_func = np.min(func)

            # find and adapt new optimal positions
            particle_positions[:, (func_prev >= func)] = particle_positions[:, (func_prev >= func)]
            func = np.min(np.array([func_prev, func]), axis=0)
            best_position = particle_positions[:, np.argmin(func)]
            min_func = np.min(func)

            # assure progress
            change = min_func_prev - min_func

            i += 1

        optimized_params = particle_positions
        value = func

        return optimized_params, value

    def run_optimizer(self, f, parameters, constraints=None):
        """
        Parameters
        ----------
        f : Parametrization.parametrization.Parametrization object
            Objective function
        parameters : numpy array / tensorflow Variable / torch tensor
            Parameters that are going to be optimized (extracted from ff_optimizable)
        constraints :
            Optimization constraints, default = None

        returns:
            optimized_params : np.array containing the optimized force field params
            value : float value of self.f
        """

        self.f = f
        self.parameters = parameters
        self.contsraints = constraints

        print("###################################################################################")
        print(
            "#                         STARTING " + self.opt_method.upper() + " OPTIMIZER                            #")
        print("###################################################################################")

        if self.opt_method == "scipy":

            optimized_params = self.optimize_with_scipy()

        elif self.opt_method == "tf_adam":

            optimized_params, value = self.optimize_with_tf_adam()

        elif self.opt_method == "pt_adam":

            optimized_params, value = self.optimize_with_pt_adam()

        elif self.opt_method == "pt_lbfgs":

            optimized_params, value = self.optimize_with_pt_lbfgs()

        elif self.opt_method == 'pso':

            optimized_params, value = self.optimize_with_pso()

        print("###################################################################################")
        print(
            "#                         " + self.opt_method.upper() + " OPTIMIZER FINISHED                            #")
        print("###################################################################################")

        return optimized_params, value






