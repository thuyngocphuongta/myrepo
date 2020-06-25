import numpy as np

class LinearLeastSquaresModel:

    """linear system solved using linear least squares
    This class serves as an example that fulfills the model interface
    needed by the ransac() function.

    """

    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = linalg.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def fit_with_least_squares(X, y):
    """
    Fits model for a given data using least squares.
    X should be an mxn matrix, where m is number of samples, and n is number of independent variables.
    y should be an mx1 vector of dependent variables.
    """
    b = np.ones((X.shape[0], 1)) # 3x1
    A = np.hstack((X, b))
    theta = np.linalg.lstsq(A, y,rcond= None)[0]
    return theta

def evaluate_model(X, y, theta, inlier_threshold):
    """
    Evaluates model and returns total number of inliers.
    X should be an mxn matrix, where m is number of samples, and n is number of independent variables.
    y should be an mx1 vector of dependent variables.
    theta should be an (n+1)x1 vector of model parameters.
    inlier_threshold should be a scalar.
    """
    b = np.ones((X.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    A = np.hstack((y, X, b)) # concat 3x6
    theta = np.insert(theta, 0, -1.)

    distances = np.abs(np.sum(A * theta, axis=1)) / np.sqrt(np.sum(np.power(theta[:-1], 2)))
    inliers = distances <= inlier_threshold
    num_inliers = np.count_nonzero(inliers == True)

    return num_inliers

def ransac(X, y, max_iters=100, samples_to_fit=2, inlier_threshold=0.1, min_inliers=10):
    best_model = None
    best_model_performance = 0

    num_samples = X.shape[0]

    for i in range(max_iters):
        sample = np.random.choice(num_samples, size=samples_to_fit, replace=False)
        model_params = fit_with_least_squares(X[sample], y[sample])
        model_performance = evaluate_model(X, y, model_params, inlier_threshold)

        if model_performance < min_inliers:
            continue

        if model_performance > best_model_performance:
            best_model = model_params
            best_model_performance = model_performance

    return best_model, best_model_performance
def test():
    # generate perfect input data

    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # the model
    B_exact = np.dot(A_exact, perfect_fit)
    assert B_exact.shape == (n_samples, n_outputs)

    # add a little gaussian noise (linear least squares alone should handle this well)
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        # add some outliers
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        non_outlier_idxs = all_idxs[n_outliers:]
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # setup model

    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)  # the first columns of the array
    output_columns = [n_inputs + i for i in range(n_outputs)]  # the last columns of the array
    debug = False
    model = LinearLeastSquaresModel(input_columns, output_columns, debug=debug)

    # run RANSAC algorithm
    inliers, outliers = ransac(all_data, model)
    return inliers, outliers
print(test())