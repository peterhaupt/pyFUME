import copy
import numpy as np
from scipy.optimize import curve_fit
import numpy.matlib
from deap import base, creator, tools, algorithms
from pyswarm import pso
import pdb

class ConsequentEstimator(object):
    """
        Creates a new consequent estimator object.
        
        Args:
            x_train: The input data.
            y_train: The output data (true label/golden standard).
            firing_strengths: Matrix containing the degree to which each rule 
                fires for each data instance.
    """

    def __init__(self, x_train, y_train, firing_strengths, categorical_indices=None):
        self.x_train = copy.deepcopy(x_train)
        self.categorical_indices = [] if categorical_indices is None else categorical_indices
        self.y_train = y_train
        self.firing_strengths = firing_strengths

    def zero_order(self, method="normalized_means"):
        """
            Estimates the consequent parameters of the zero-order Sugeno-Takagi model using one of the following methods:
                - normalized means (normalized_means)
                - global Least Squares Estimation (global_LSE)
                - local Least Squares Estimation (local_LSE)
                - gradient descent (gradient_descent)
                - genetic algorithm (GA)
                - particle swarm optimization (PSO)

        
            Args:
                df: default value returned when the sum of grades equals to one (default = 0).
        
            Returns:
                The parameters for the consequent function.
        """
        if method == "normalized_means":
            p = np.zeros((self.firing_strengths.shape[1]))
            for clus in range(0, self.firing_strengths.shape[1]):
                fs = self.firing_strengths[:, clus]
                fs = np.fmax(fs, np.finfo(np.float64).eps)  # avoid 0's in the matrix
                normalized_weights = fs / fs.sum(0)
                s = np.multiply(normalized_weights, self.y_train)
                p[clus] = sum(s)
            
            # ESTIMATE CONSEQUENTS - normalized means - Table 38
            # print("The parameters for the consequent function are stored in p.")
            # pdb.set_trace()
            
            return p
        
        elif method == "global_LSE":
            try:
                # Number of rules (clusters)
                num_rules = self.firing_strengths.shape[1]

                # Ensure the matrix is not singular by adding a small value (regularization)
                epsilon = np.finfo(np.float64).eps

                # Calculate the sum of the degree of fulfillment (DOF) for each data point
                sumDOF = np.sum(self.firing_strengths, axis=1)
                sumDOF[sumDOF == 0] = 1  # Avoid division by zero for data points with no applicable rules

                # Normalize the firing strengths by the sum of DOF
                normalized_firing_strengths = self.firing_strengths / sumDOF[:, np.newaxis]

                # Perform Least Squares Estimation (LSE)
                # This solves the normal equation: p = (F.T @ F)^(-1) @ F.T @ y
                F = normalized_firing_strengths
                FtF_inv = np.linalg.inv(F.T @ F + epsilon * np.eye(num_rules))  # Regularization
                p = FtF_inv @ F.T @ self.y_train

                # ESTIMATE CONSEQUENTS - global LSE - Table 39
                # print("The parameters for the consequent function are stored in p.")
                # pdb.set_trace()

                return p
            
            except np.linalg.LinAlgError:
                # Fall back to normalized_means if global_LSE fails due to singular matrix
                print("Singular matrix encountered in global_LSE, falling back to normalized_means.")
                return self.zero_order(method="normalized_means")
        
        elif method =="local_LSE":
            # Number of rules (clusters)
            num_rules = self.firing_strengths.shape[1]
            
            # Number of data points (rows in x_train)
            num_samples = self.x_train.shape[0]

            # Pre-allocate variable for the consequent parameters
            p = np.zeros(num_rules)

            # Loop through each rule (local least squares estimation per rule)
            for i in range(num_rules):
                # Select firing strength for the current rule
                firing_strengths_rule = self.firing_strengths[:, i]
                
                # Avoid division by zero: if all firing strengths are zero, skip this rule
                if np.sum(firing_strengths_rule) == 0:
                    continue
                
                # Weight the input (x_train) and the output (y_train) using the firing strengths
                # In a zero-order model, we only care about weighting the outputs (since x doesn't matter)
                weighted_y = self.y_train * np.sqrt(firing_strengths_rule)
                
                # In a zero-order model, we estimate a constant for each rule. 
                # So, we're effectively summing the weighted output and normalizing by the firing strengths.
                p[i] = np.sum(weighted_y) / np.sum(firing_strengths_rule)
    
            # ESTIMATE CONSEQUENTS - local LSE - Table 40
            # print("The parameters for the consequent function are stored in p.")
            # pdb.set_trace()
            
            return p
        
        elif method == "gradient_descent":

            learning_rate=0.01
            iterations=1000
            
            # Number of rules (clusters)
            num_rules = self.firing_strengths.shape[1]
            
            # Initialize the parameters (consequents) with small random values or zeros
            p = np.random.randn(num_rules) * 0.01  # Small random initialization

            # Loop over the number of iterations
            for iter in range(iterations):
                # Calculate the predictions
                y_pred = np.dot(self.firing_strengths, p)  # weighted sum for all rules
                
                # Calculate the gradient for each rule (p_k)
                gradients = np.zeros(num_rules)
                for k in range(num_rules):
                    # Gradient of MSE with respect to p_k
                    gradients[k] = (-2 / len(self.y_train)) * np.dot(self.firing_strengths[:, k], (self.y_train - y_pred))

                # Update the parameters (consequents) using gradient descent
                p -= learning_rate * gradients
                
                # Optionally, monitor the MSE during training
                mse = np.mean((self.y_train - y_pred) ** 2)
                if iter % 100 == 0:
                    print(f"Iteration {iter}: MSE = {mse}")

            # ESTIMATE CONSEQUENTS - gradient descent - Table 41
            # print("The parameters for the consequent function are stored in p.")
            # pdb.set_trace()
            
            return p
        
        elif method == "GA":
            # define the genetic algorithm parameters
            n_generations=100
            population_size=50
            cxpb=0.5
            mutpb=0.2

            # Number of rules (clusters)
            num_rules = self.firing_strengths.shape[1]
            
            # Define the fitness function as minimizing the MSE
            def mse_fitness(individual):
                # Calculate predictions using the individual's consequent parameters
                y_pred = np.dot(self.firing_strengths, individual)
                # Calculate mean squared error (MSE)
                mse = np.mean((self.y_train - y_pred) ** 2)
                return mse,

            # Set up the Genetic Algorithm
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize MSE
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            # Initialize the toolbox for the genetic algorithm
            toolbox = base.Toolbox()

            # Define individual (a vector of size num_rules with random initialization)
            toolbox.register("attr_float", np.random.uniform, -1, 1)  # Random between -1 and 1
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_rules)
            
            # Define the population (a list of individuals)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Register the evaluation, crossover, mutation, and selection functions
            toolbox.register("evaluate", mse_fitness)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Create the population
            population = toolbox.population(n=population_size)

            # Apply the genetic algorithm
            result_population, _ = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, verbose=True)

            # Extract the best individual from the population
            best_individual = tools.selBest(result_population, k=1)[0]

            # Convert the best individual to a NumPy array and round
            best_individual = np.array(best_individual).round(decimals=2)

            # ESTIMATE CONSEQUENTS - GA - Table 42
            # print("The parameters for the consequent function are stored in best_individual.")
            # pdb.set_trace()
            
            return best_individual  # Best parameters for the consequent functions
        
        elif method == "PSO":

            # Number of rules (clusters)
            num_rules = self.firing_strengths.shape[1]
            
            # Define the fitness function as minimizing the MSE
            def mse_fitness(p):
                # Calculate predictions using the particle's consequent parameters
                y_pred = np.dot(self.firing_strengths, p)
                # Calculate mean squared error (MSE)
                mse = np.mean((self.y_train - y_pred) ** 2)
                return mse

            # Define lower and upper bounds for the particle swarm optimization
            lb = [-1] * num_rules  # Lower bounds for the consequents
            ub = [1] * num_rules   # Upper bounds for the consequents

            # Run PSO
            best_consequents, best_mse = pso(mse_fitness, lb, ub, swarmsize=50, maxiter=100)
            
            # ESTIMATE CONSEQUENTS - particle swarm optimization - Table 43
            print("The parameters for the consequent function are stored in best_consequents.")
            pdb.set_trace()

            return best_consequents  # Best parameters for the consequent functions

    def suglms(self, global_fit=False, df=0):
        """
            Estimates the consequent parameters in the first-order Sugeno-Takagi model using least squares.
        
            Args:
                global_fit: Use the local (global_fit=False) or global (global_fit=True) least mean squares estimates. Global estimates functionality is still in beta mode, so use with caution.
                df: default value returned when the sum of grades equals to one (default = 0).
        
            Returns:
                The parameters for the consequent function.
        """

        x = self.x_train.copy()
        y = self.y_train.copy()
        f = self.firing_strengths.copy()

        # pdb.set_trace()
        # ONE HOT ENCODE CATEGORICAL VARIABLES
        slices = []
        for i in range(x.shape[1]):
            if i in self.categorical_indices:
                # categorical variables are expected to be INTEGERS in the range 0,...,n
                col = x[:, i].astype(np.uint32)
                one_hot = np.zeros((x.shape[0], col.max() + 1))
                one_hot[np.arange(x.shape[0]), col] = 1
                slices.append(one_hot[:, :-1])
            else:
                slices.append(x[:, i].reshape(-1, 1))
        x = np.concatenate(slices, axis=1)

        # Check if input X contains one column of ones (for the constant). If not, add it.
        u = np.unique(x[:, -1])
        if u.shape[0] != 1 or u[0] != 1:
            x = np.hstack((x, np.ones((x.shape[0], 1))))

        # attempt to get a working zero order implementation - but not used anymore
        # Get the number of columns in x
        num_cols = x.shape[1]
        # Set everything except the last column to 0
        x[:, :num_cols-1] = 0


        # Find the number of data points (mx & mf) , the number of variables (nx) and the
        # number of clusters (nf) 
        mx, nx = x.shape
        mf, nf = f.shape

        # Calculate the sum of the degree of fulfillement (DOF) for each data point
        sumDOF = np.sum(f, 1)

        # ESTIMATE CONSEQUENTS - Table 17
        # print("The sum of the degree of fulfillment (DOF) is stored in sumDOF.")
        # pdb.set_trace()

        # When degree of fulfillment is zero (which means no rule is applicable), set to one
        NoRule = sumDOF == 0
        sumDOF[NoRule] = 1
        sumDOF = np.matlib.repmat(sumDOF, nf, 1).T

        if nf == 1:
            global_fit = False

        if global_fit is True:  # Global least mean squares estimates

            # Still under construction!

            # Auxillary variables
            f1 = x.flatten()
            s = np.matlib.repmat(f1, nf, 1).T
            xx = np.reshape(s, (nx, nf * mx), order='F')
            s = xx.T
            x1 = np.reshape(s, (mx, nf * nx), order='F')
            x = x.T  # reshape data matrix

            # (reshaped) vector of f devided by the sum of each row of f
            # (normalised membership degree)
            xx = (f.T.flatten() / sumDOF.T.flatten())

            # reshape partition matrix
            s = np.matlib.repmat(xx, nx, 1).T
            f1 = np.reshape(s, (mx, nf * nx), order='F')

            # CHANGE HERE!!!
            # f1 - betas
            # x1 - input
            x1 = f1 * x1 # removed x1

            # Find least squares solution
            #            xp = np.linalg.lstsq(x1,y,rcond=None)

            # Perform QR decomposition
            Q, R = np.linalg.qr(x1)  # qr decomposition of A
            Qy = np.dot(Q.T, y)  # computing Q^T*b (project b onto the range of A)
            xx = np.linalg.solve(R, Qy)

            p = np.reshape(xx, (nf, nx), order='F')

            # Local models
            yl = np.transpose(x).dot(np.transpose(p))  #

            # Global model
            ym = x1.dot(p.flatten()) + df * NoRule
            ylm = yl.copy()

            # Mask all memberships < 0.2 with NaN's for plots
            ylm[f < 0.2] = np.NaN

        elif global_fit is False:  # local weighted least mean squares estimates
            # Pre-allocate variable
            p = np.zeros([nf, nx])

            for i in range(0, nf):
                # Select firing strength of the selected rule
                w = f[:, i]

                # Weight input with firing strength
                xw = x * np.sqrt(w[:, np.newaxis])

                # Weight output with firing strength
                yw = y * np.sqrt(w)

                # Perform least squares with weighted input and output
                prm, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
                p[i] = prm
                # ESTIMATE CONSEQUENTS - Table 17
                # print("The weighted input is stored in xw.")
                # print("The weighted output is stored in yw.")
                # print("The least squares solution is stored in prm.")
                # print("The parameters for the consequent function are stored in p.")
                # pdb.set_trace()

        return p  # ,ym,yl,ylm
