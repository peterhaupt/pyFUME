import numpy as np
from collections import Counter
from simpful import SingletonsSet
from scipy.optimize import curve_fit
from numpy import linspace, array
from collections import defaultdict
import pdb


def is_complete(G):
    nodelist = G.nodes
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n * (n - 1) / 2


class AntecedentEstimator(object):
    """
        Creates a new antecedent estimator object.
        
        Args:
            x_train: The input data.
            partition_matrix: The partition matrix of the input data (generated 
                by a clustering the data).
    """

    def __init__(self, x_train, partition_matrix):
        self.xtrain = x_train
        self.partition_matrix = partition_matrix
        self._info_for_simplification = None
        self._calculate_all_extreme_values()
        # ESTIMATE ANTECENDENTS - determine extreme values - Table 11
        # print("The extreme values are stored in self._extreme_values. The first element is the minimum and the second element is the maximum for each variable.")
        # pdb.set_trace()
        self._setnes_removed_sets = defaultdict(list)

    def determineMF(self, mf_shape='gauss', merge_threshold=1.0, categorical_indices=None, setnes_threshold=1.0, global_singleton=None):
        """
            Estimates the parameters of the membership functions that are used 
            as antecedents of the fuzzy rules.
            
            Args:
                mf_shape: The desired shape of the fuzzy sets. The user can choose
                    from 'gauss' (gaussian), 'gauss2' (double gaussion) or 'sigmoid' 
                    (sigmoidal) (default = gauss).
                merge_threshold: Threshold for the merging of fuzzy sets for 
                    the GRABS approach. By default no merging takes place 
                    (default = 1.0).
            Returns:
                A list with the estimated parameters of the membership functions (format if gauss: mu, sigma; if gauss2: mu1, sigma1, mu2, sigma2)
        """
        mf_list = []

        # mf_list is structured as follows:
        # - an outer list of V variables
        # - each item of the outer list contains C fuzzy set, one for each cluster
        number_of_variables = self.xtrain.shape[1]
        for i in range(0, number_of_variables):
            if categorical_indices is not None and i in categorical_indices:
                if global_singleton is not None:
                    unique_values = np.unique(self.xtrain[:, i])
                    if global_singleton == 'softmax':
                        from scipy.special import softmax
                        pm = softmax(self.partition_matrix, axis=0)
                    else:
                        pm = self.partition_matrix
                    pml = []
                    for uv in unique_values:
                        pmn = pm[self.xtrain[:, i] == uv].sum(axis=0)
                        pml.append(list(pmn / pmn.sum()))
                    sets = list(zip(*pml))
                    for s in sets:
                        mf_list.append(('singleton', list(zip(unique_values, s))))
                else:
                    xin = self.xtrain[:, i]
                    cluster_frequencies_counter = [[] for _ in range(0, self.partition_matrix.shape[1])]
                    unique_values = set()
                    for j in range(0, self.partition_matrix.shape[0]):
                        cl = np.argmax(self.partition_matrix[j, :])  # Determine the cluster the instance belongs to the most
                        unique_values.add(xin[j])
                        cluster_frequencies_counter[cl].append(xin[j])
                    cluster_frequencies = {}
                    value_frequencies = {k: [] for k in unique_values}  # used to force the sum for each variable to 1
                    for j, clf in enumerate(cluster_frequencies_counter):
                        total_number = len(clf)
                        counter = Counter(clf)
                        # k is the element of the universe of discourse and n / total_number is the membership function
                        tmp_dict = {}
                        for k, n in counter.items():
                            freq = n / total_number
                            tmp_dict[k] = freq
                            value_frequencies[k].append(freq)
                        cluster_frequencies[j] = tmp_dict
                    # Force sum to 1 for each value of the categorical feature
                    # Computing the total sum for each value
                    total_sums = {k: sum(value_frequencies[k]) for k in unique_values}
                    for j in cluster_frequencies:
                        mfs = []
                        for k in unique_values:
                            if k in cluster_frequencies[j]:
                                mfs.append((k, cluster_frequencies[j][k] / total_sums[k]))
                            else:
                                mfs.append((k, 0.0))
                        prm = ('singleton', mfs)
                        mf_list.append(prm)
            else:
                xin = self.xtrain[:, i]
                if all(y in (0, 1) for y in xin):  # Add noise to binary variables
                    noise = np.random.normal(0, 0.001, xin.shape[0])
                    xin = xin + noise

                for j in range(0, self.partition_matrix.shape[1]):
                    mfin = self.partition_matrix[:, j]
                    mf, xx = self._convexMF(xin=xin, mfin=mfin)

                    #############
                    ###### CONTINUE HERE TOMORROW
                    prm = self._fitMF(x=xx, mf=mf, mf_shape=mf_shape)
                    mf_list.append(prm)

        if merge_threshold < 1.0 or setnes_threshold < 1.0:
            self._check_similarities(mf_list, number_of_variables, threshold=merge_threshold, setnes_threshold=setnes_threshold)

        # print(self._info_for_simplification)

        return mf_list

    def is_subclique(self, G, nodelist):
        H = G.subgraph(nodelist)
        n = len(nodelist)
        return H.size() == n * (n - 1) / 2

    def _extreme_values_for_variable(self, v):
        return min(self.xtrain.T[v]), max(self.xtrain.T[v])

    def _calculate_all_extreme_values(self):
        num_variables = len(self.xtrain.T)
        self._extreme_values = [self._extreme_values_for_variable(v) for v in range(num_variables)]

    def _check_similarities(self, mf_list, number_of_variables,
            approx_points=100,
            threshold=1.,
            setnes_threshold=1.,
            verbose=True
            ):

        number_of_clusters = len(mf_list) // number_of_variables

        from collections import defaultdict

        things_to_be_removed = defaultdict(list)

        """ 
            This function assesses the pair-wise similarities between 
            the clusters mapped on each variable.
            It returns a dictionary of this kind:
            variable -> list of similar couples for that variable + jaccard sim

        """

        for v in range(number_of_variables):

            if verbose:
                print(" * Trying to simplify variable", v)

            mi, ma = self._extreme_values_for_variable(v)
            points = np.linspace(mi, ma, approx_points)

            for c1 in range(number_of_clusters):
                for c2 in range(c1 + 1, number_of_clusters):

                    index1 = v * number_of_clusters + c1
                    index2 = v * number_of_clusters + c2
                    funname1, params1 = mf_list[index1]
                    funname2, params2 = mf_list[index2]

                    if funname1 == "gauss":

                        from numpy import linspace, array

                        first_cluster = array([self._gaussmf(x, params1[0], params1[1]) for x in points])
                        second_cluster = array([self._gaussmf(x, params2[0], params2[1]) for x in points])

                        intersection = sum([min(x, y) for x, y in zip(first_cluster, second_cluster)])
                        union = sum([max(x, y) for x, y in zip(first_cluster, second_cluster)])

                        jaccardsim = (intersection / union)

                        if jaccardsim > threshold:
                            things_to_be_removed[v].append([c1, c2, jaccardsim])

                            # print("%.2f is fine" % jaccardsim)

                    else:
                        raise Exception("Not implemented yet")

                # Setnes' rule simplification: detect which sets are similar to the universal set
                #                              using Jaccard similarity and a threshold.
                if setnes_threshold<1.:
                    index1 = v*number_of_clusters + c1
                    funname1, params1 = mf_list[index1]

                    if funname1== "gauss":

                        first_cluster = array([self._gaussmf(x, params1[0], params1[1]) for x in points])
                        second_cluster = array([1 for x in points]) # universal set

                        intersection = sum([min(x,y) for x,y in zip(first_cluster, second_cluster)])
                        union        = sum([max(x,y) for x,y in zip(first_cluster, second_cluster)])

                        jaccardsim = (intersection/union)

                        if jaccardsim>setnes_threshold:
                            self._setnes_removed_sets[v].append(c1)
                            print (" * Variable %d, cluster %d is too similar to universal set (threshold: %.2f): marked for removal" % (v,c1+1,setnes_threshold))

                    else:
                        raise Exception("Setnes' simplification for non-Gaussian functions not implemented yet")



        #for k,v in things_to_be_removed.items():            print (k, v)
        #exit()
        """
            This function must return a dictionary of items like:
            (variable, cluster) -> mapped_cluster
        """

        self._info_for_simplification = {}
        for var_num, value in things_to_be_removed.items():

            subcomponents = self._create_graph(value)

            for subcomp in subcomponents:
                # print (is_complete(subcomp))

                if is_complete(subcomp):
                    retained = list(subcomp.nodes())[0]
                    # print ("retain: %d" % retained)
                    for el in list(subcomp.nodes()):
                        if el != retained:
                            self._info_for_simplification[(var_num, el)] = retained

        dropped_stuff = self.get_number_of_dropped_fuzzy_sets()
        print (" * %d antecedent clauses will be simplified using a threshold %.2f" % (dropped_stuff, threshold))
        if verbose:
            print(" * GRABS remapping info:", self._info_for_simplification)
            print(" * Setnes simplification dictionary variable ==> list of clusters/fuzzy sets to be removed:", self._setnes_removed_sets)

        self._info_for_simplification

    def get_number_of_dropped_fuzzy_sets(self):
        return len(self._info_for_simplification)

    def _create_graph(self, list_of_arcs):
        from networkx import Graph, connected_components
        G = Graph()
        nodelist = []
        for arc in list_of_arcs:
            G.add_edge(arc[0], arc[1])
            nodelist.append(arc[0])
            nodelist.append(arc[1])
        S = [G.subgraph(c).copy() for c in connected_components(G)]
        return S

    def _convexMF(self, xin, mfin, norm=1, nc=1000): 

        # Calculates the convex membership function that envelopes a given set of
        # data points and their corresponding membership values. 

        # Input:
        # Xin: N x 1 input domain (column vector)
        # MFin: N x 1 corresponding membership values 
        # nc: number of alpha cut values to consider (default=101)
        # norm: optional normalization flag (0: do not normalize, 1 : normalize, 
        # default=1)
        #
        # Output:
        # mf: membership values of convex function
        # x: output domain values    

        # Normalize the membership values (if requested)
        if norm == 1:
            mfin = np.divide(mfin, np.max(mfin))

        # ESTIMATE ANTECENDENTS - section 3.2. - normalise the membership values of fuzzy c-means clustering - Table 12
        # print("The normalise membership values are stored in mfin. The input values of the feature are stored in xin.")
        # pdb.set_trace()

        # Initialize auxilary variables
        acut = np.linspace(0, np.max(mfin), nc)
        mf = np.full(2 * nc, np.nan)
        x = np.full(2 * nc, np.nan)

        if np.any(mfin > 0):
            x[0] = np.min(xin[mfin > 0])
            x[nc] = np.max(xin[mfin > 0])
            mf[0] = 0
            mf[nc] = 0

        # Determine the elements in the alpha cuts
        for i in range(0, nc):
            if np.any(mfin > acut[i]):
                x[i] = np.min(xin[mfin > acut[i]])
                x[i + nc] = np.max(xin[mfin > acut[i]])
                mf[i] = acut[i]
                mf[i + nc] = acut[i]

        # ESTIMATE ANTECENDENTS - determine elements in the alpha cuts - first part of section 3.2
        # print("The input values of the feature are stored in x and the membership values are stored in mf.")
        # print("The first half of the values are the minimum and the second half are the maximum for each alpha cut.")
        # pdb.set_trace()

        # # Determine the elements in the alpha cuts    
        # for i in range(0,nc):
        #     tmp1 = mfin>acut[i]
        #     if len(tmp1)==0:
        #         tmp=xin[tmp1]
        #         np.sort(tmp)
        #         x[i]=tmp[0]
        #         x[i+nc]=tmp[-1]
        #         mf[i]=acut[i]
        #         mf[i+nc]=acut[i]

        # Delete NaNs
        idx = np.isnan(x)
        x = x[idx == False]
        mf = mf[idx == False]

        # ESTIMATE ANTECENDENTS - delete NaN values - second part of section 3.2
        # print("The cleaned input values of the feature are stored in x and the membership values are stored in mf.")
        # pdb.set_trace()

        # Sort vectors based on membership value (descending order)
        indmf = mf.argsort(axis=0)
        indmf = np.flipud(indmf)
        mf = mf[indmf]
        x = x[indmf]

        # ESTIMATE ANTECENDENTS - sort vectors descending - third part of section 3.2
        # print("The sorted input values of the feature are stored in x and the membership values are stored in mf.")
        # pdb.set_trace()

        # Find duplicate values for x and onlykeep the ones with the highest membership value
        _, ind = np.unique(x, return_index=True, return_inverse=False, return_counts=False, axis=None)
        mf = mf[ind]
        x = x[ind]
        # ESTIMATE ANTECENDENTS - find duplicates - fourth part of section 3.2
        # print("The input values of the feature without duplicates are stored in x and the membership values are stored in mf.")
        # pdb.set_trace()

        # Sort vectors based on x value (ascending order)
        indx = x.argsort(axis=0)
        mf = mf[indx]
        x = x[indx]

        # ESTIMATE ANTECENDENTS - sort vectors ascending - fifth part of section 3.2
        # print("The input values of the feature in ascending order are stored in x and the membership values are stored in mf.")
        # pdb.set_trace()

        xval = np.linspace(np.min(x), np.max(x), nc)
        mf = np.interp(xval, x, mf, left=None, right=None, period=None)
        x = xval

        # ESTIMATE ANTECENDENTS - determine final membership values - sixth part of section 3.2
        # print("Final input values of the feature are stored in x and the membership values are stored in mf.")
        # pdb.set_trace()

        return mf, x

    def _fitMF(self, x, mf, mf_shape='gauss'):
        # Fits parametrized membership functions to a set of pointwise defined 
        # membership values.
        #
        # Input:
        # x:  N x 1 domain of input variable
        # mf: N x 1 membership values for input data x 
        # shape: Type of membership function to fit (possible values: 'gauss', 
        # 'gauss2' and 'sigmoid')
        #
        # Output:
        # param: matrix of membership function parameters

        if mf_shape == 'gauss':
            # Determine initial parameters
            mu = sum(x * mf) / sum(mf)
            # ESTIMATE ANTECENDENTS - determine initial parameters for parametrized membership functions - Table 13
            # print("The initial paramters for the mean (mu) are stored in mu.")
            # pdb.set_trace()
            mf[mf == 0] = np.finfo(np.float64).eps
            sig = np.mean(np.sqrt(-((x - mu) ** 2) / (2 * np.log(mf))))
            # ESTIMATE ANTECENDENTS - determine initial parameters for parametrized membership functions - Table 14
            # print("The initial paramters for the standard deviation are stored in sig.")
            # pdb.set_trace()

            # Fit parameters to the data using least squares
            #            print('mu=', mu, 'sig=', sig)
            param, _ = curve_fit(self._gaussmf, x, mf, p0=[mu, sig], bounds=((-np.inf, 0), (np.inf, np.inf)),
                                 maxfev=10000)
            
            # ESTIMATE ANTECENDENTS - determine optimal parameters for parametrized membership functions - Table 15
            # print("The optimal parameters are stored in param.")
            # pdb.set_trace()


        elif mf_shape == 'gauss2':
            try:
                # Attempt to fit 'gauss2' MF
                mu1 = x[mf >= 0.95][0]  # Leftmost high membership point
                mu2 = x[mf >= 0.95][-1]  # Rightmost high membership point
                xmf = x[mf >= 0.5]  # Mid-range membership points
                sig1 = (mu1 - xmf[0]) / (np.sqrt(2 * np.log(2)))  # Standard deviation on the left
                sig2 = (xmf[-1] - mu2) / (np.sqrt(2 * np.log(2)))  # Standard deviation on the right
                sig1 = sig1 if sig1 != 0 else 0.1  # Avoid zero variance
                sig2 = sig2 if sig2 != 0 else 0.1  # Avoid zero variance

                # Fit 'gauss2' parameters
                param, _ = curve_fit(self._gauss2mf, x, mf, p0=[mu1, sig1, mu2, sig2], maxfev=1000,
                                    bounds=((-np.inf, 0, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf)))
            except (IndexError, RuntimeError):
                # Fall back to 'gauss' if 'gauss2' fails
                print("Falling back to 'gauss' from 'gauss2' due to fitting issues.")
                # Determine initial parameters for Gaussian MF
                mu = sum(x * mf) / sum(mf)
                mf[mf == 0] = np.finfo(np.float64).eps  # To avoid log(0)
                sig = np.mean(np.sqrt(-((x - mu) ** 2) / (2 * np.log(mf))))

                # Fit Gaussian parameters to data
                param, _ = curve_fit(self._gaussmf, x, mf, p0=[mu, sig], bounds=((-np.inf, 0), (np.inf, np.inf)), maxfev=10000)

        elif mf_shape == 'sigmoid':
            # Try fitting the sigmoidal membership function
            try:
                if np.argmax(mf) - np.argmin(mf) > 0:  # Sloping to the right
                    c = x[mf >= 0.5][0] if len(x[mf >= 0.5]) > 0 else x[0]
                    s = 1
                else:  # Sloping to the left
                    c = x[mf <= 0.5][0] if len(x[mf <= 0.5]) > 0 else x[-1]
                    s = 1

                param, _ = curve_fit(self._sigmoid, x, mf, p0=[c, s], maxfev=1000)

            except RuntimeError:
                print('Failed to fit sigmoidal membership function, falling back to Gaussian.')
                # Fallback to Gaussian if sigmoidal fitting fails
                mf_shape = 'gauss'
                mu = sum(x * mf) / sum(mf)
                mf[mf == 0] = np.finfo(np.float64).eps
                sig = np.mean(np.sqrt(-((x - mu) ** 2) / (2 * np.log(mf))))
                
                param, _ = curve_fit(self._gaussmf, x, mf, p0=[mu, sig], 
                                    bounds=((-np.inf, 0), (np.inf, np.inf)), maxfev=10000)
                
        elif mf_shape == 'invgauss':
            # Determine initial parameters for inverse Gaussian
            mu = sum(x * mf) / sum(mf)
            lambda_param = np.var(x)  # Initialize shape parameter (λ) with variance
            
            # Fit parameters to the data using least squares for inverse Gaussian
            param, _ = curve_fit(self._invgaussmf, x, mf, p0=[mu, lambda_param], maxfev=1000,
                                bounds=((0, 0), (np.inf, np.inf)))
            
        elif mf_shape == 'trimf':
            # Estimate initial parameters for triangular MF
            a = np.min(x[mf >= 0.1])  # Left endpoint
            b = x[np.argmax(mf)]      # Peak of the triangle
            c = np.max(x[mf >= 0.1])  # Right endpoint

            # Fit parameters to the data using least squares for triangular MF
            param, _ = curve_fit(self._trimf, x, mf, p0=[a, b, c], maxfev=10000,
                                bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))
            
            # Ensure the parameters follow the order a <= b <= c
            param = np.sort(param)

        elif mf_shape == 'trapmf':
            # Estimate initial parameters for trapezoidal MF
            a = np.min(x[mf >= 0.1])  # Left foot
            b = np.min(x[mf >= 0.5])  # Left shoulder
            c = np.max(x[mf >= 0.5])  # Right shoulder
            d = np.max(x[mf >= 0.1])  # Right foot

            # Fit parameters to the data using least squares for trapezoidal MF
            param, _ = curve_fit(self._trapmf, x, mf, p0=[a, b, c, d], maxfev=10000,
                                bounds=((-np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf)))

            # Ensure the parameters follow the order a <= b <= c <= d
            param = np.sort(param)

        elif mf_shape == 'invsigmoid':
            try:
                if np.argmax(mf) - np.argmin(mf) > 0:  # Sloping to the right
                    c = x[mf >= 0.5][0] if len(x[mf >= 0.5]) > 0 else x[0]
                    s = -1  # Negative slope for inverse sigmoid
                else:  # Sloping to the left
                    c = x[mf <= 0.5][0] if len(x[mf <= 0.5]) > 0 else x[-1]
                    s = -1

                param, _ = curve_fit(self._invsigmoid, x, mf, p0=[c, s], maxfev=1000)

            except RuntimeError:
                print('Failed to fit inverse sigmoidal membership function, falling back to Gaussian.')
                # Fallback to Gaussian if inverse sigmoid fitting fails
                mf_shape = 'gauss'
                mu = sum(x * mf) / sum(mf)
                mf[mf == 0] = np.finfo(np.float64).eps
                sig = np.mean(np.sqrt(-((x - mu) ** 2) / (2 * np.log(mf))))
                
                param, _ = curve_fit(self._gaussmf, x, mf, p0=[mu, sig], 
                                    bounds=((-np.inf, 0), (np.inf, np.inf)), maxfev=10000)
                
        elif mf_shape == 'invgauss':
            # Estimate initial parameters for inverse Gaussian MF
            mu = sum(x * mf) / sum(mf)
            lambda_param = np.var(x)  # Initialize shape parameter (λ) with variance
            
            # Fit parameters to the data using least squares for inverse Gaussian MF
            param, _ = curve_fit(self._invgaussian, x, mf, p0=[mu, lambda_param], maxfev=1000,
                                bounds=((0, 0), (np.inf, np.inf)))
            
        elif mf_shape == 'crisp':
            # For crisp membership functions, we only need the left and right extremes
            a = np.min(x[mf >= 0.5])  # Left extreme value
            b = np.max(x[mf >= 0.5])  # Right extreme value
            
            # Return the parameters directly, no fitting required
            param = [a, b]

        return mf_shape, param

    def _gaussmf(self, x, mu, sigma, a=1):
        # x:  (1D array)
        # mu: Center of the bell curve (float)
        # sigma: Width of the bell curve (float)
        # a: normalizes the bell curve, for normal fuzzy set a=1 (float) 
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def _gauss2mf(self, x, mu1, sigma1, mu2, sigma2):
        # x: Data
        # mu1: Center of the leftside bell curve
        # sigma1: Standard deviation that determines the width of the leftside bell curve
        # mu2: Center of the rightside bell curve
        # sigma2: Standard deviation that determines the width of the rightside bell curve
        y = np.ones(len(x))
        idx1 = x <= mu1
        idx2 = x > mu2
        y[idx1] = self._gaussmf(x[idx1], mu1, sigma1)
        y[idx2] = self._gaussmf(x[idx2], mu2, sigma2)
        return y

    def _sigmoid(self, x, c, s):
        """
        Sigmoid membership function.

        Args:
            x (array): Input data.
            c (float): Controls the midpoint of the sigmoid.
            s (float): Controls the steepness of the curve.

        Returns:
            array: Membership values corresponding to input data x.
        """
        # Clip the input to avoid overflow in exp.
        exp_input = np.clip(-s * (x - c), -500, 500)

        # Return the sigmoid safely
        return 1. / (1. + np.exp(exp_input))
    
    def _invgaussmf(self, x, mu, lambda_param):
        """
        Inverse Gaussian (Wald) membership function.

        Args:
            x (array): Input data.
            mu (float): Mean of the distribution.
            lambda_param (float): Shape parameter.

        Returns:
            array: Membership values corresponding to input data x.
        """
        from numpy import exp, sqrt, pi
        
        # Avoid invalid values in the input by enforcing x > 0
        x = np.clip(x, 1e-6, None)  # Ensure x is always positive and non-zero to avoid division by zero
        
        # Avoid negative or zero values of lambda_param
        lambda_param = max(lambda_param, 1e-6)
        
        # Inverse Gaussian formula with safe values for x
        return sqrt(lambda_param / (2 * pi * x**3)) * exp(-lambda_param * (x - mu)**2 / (2 * mu**2 * x))
    
    def _trimf(self, x, a, b, c):
        """
        Triangular membership function.
        
        Args:
            x (array): Input data.
            a (float): Left endpoint of the triangle.
            b (float): Peak of the triangle.
            c (float): Right endpoint of the triangle.
            
        Returns:
            array: Membership values corresponding to input data x.
        """
        return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)
    
    def _trapmf(self, x, a, b, c, d, epsilon=1e-6):
        """
        Trapezoidal membership function with safeguards against division by zero.

        Args:
            x (array): Input data.
            a (float): Left foot of the trapezoid (start of the slope).
            b (float): Left shoulder of the trapezoid (start of the flat top).
            c (float): Right shoulder of the trapezoid (end of the flat top).
            d (float): Right foot of the trapezoid (end of the slope).
            epsilon (float): Small value to prevent division by zero.

        Returns:
            array: Membership values corresponding to input data x.
        """
        # Ensure that we don't divide by zero by adding epsilon if the denominator is too small
        denom1 = max(b - a, epsilon)
        denom2 = max(d - c, epsilon)
        
        # Calculate the trapezoidal membership function
        return np.maximum(np.minimum(np.minimum((x - a) / denom1, 1), (d - x) / denom2), 0)
    
    def _invsigmoid(self, x, c, s):
        """
        Inverse Sigmoid membership function.

        Args:
            x (array): Input data.
            c (float): Controls the midpoint of the inverse sigmoid.
            s (float): Controls the steepness of the curve.

        Returns:
            array: Membership values corresponding to input data x.
        """
        # Clip the input to avoid overflow in exp. 
        # The range (-500, 500) is chosen because np.exp(500) is still manageable in float64.
        exp_input = np.clip(s * (x - c), -500, 500)

        # Return the inverse sigmoid safely
        return 1. / (1. + np.exp(exp_input))
    
    def _invgaussian(self, x, mu, lambda_param):
        """
        Inverse Gaussian membership function (Wald distribution).

        Args:
            x (array): Input data.
            mu (float): Mean of the distribution.
            lambda_param (float): Shape parameter.
        
        Returns:
            array: Membership values corresponding to input data x.
        """
        from numpy import exp, sqrt, pi
        # Avoid invalid values in the input by enforcing x > 0
        x = np.clip(x, 1e-6, None)  # Ensure x is always positive and non-zero to avoid division by zero
        
        # Avoid negative or zero values of lambda_param
        lambda_param = max(lambda_param, 1e-6)
        
        # Inverse Gaussian formula
        return sqrt(lambda_param / (2 * pi * x**3)) * exp(-lambda_param * (x - mu)**2 / (2 * mu**2 * x))