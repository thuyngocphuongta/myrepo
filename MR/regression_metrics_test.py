#from dimensional_model_utils import *
import numpy as np
from scipy import linalg, dot

#y_true = [3, 1, 2, 7]
#y_pred_pos_proba = [2.5, 1, 2, 8]
#y_pred = [2.5, 1, 2, 8]

#print(dm_binary_classification_metrics(y_true, y_pred_pos_proba,y_pred))
def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


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


def ransac(y_true, y_pred, n, k, threshold, d, debug=False, return_all=False):
    """fit model parameters to data using the RANSAC algorithm

    This implementation written from pseudocode found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    {{{
    Given:
        data - a set of observed data points
        model - a model that can be fitted to data points
        n - the minimum number of data values required to fit the model ??
        k - the maximum number of iterations allowed in the algorithm  ??
        t - a threshold value for determining when a data point fits a model  -->  0 =< t <= 1 (step = 0.1)
        d - the number of close data values required to assert that a model fits well to data  --> d = y_true +- t
    Return:
        bestfit - model parameters which best fit the data (or nul if no good model is found)
    iterations = 0
    bestfit = nul
    besterr = something really large
    while iterations < k {
        maybeinliers = n randomly selected values from data
        maybemodel = model parameters fitted to maybeinliers
        alsoinliers = empty set
        for every point in data not in maybeinliers {
            if point fits maybemodel with an error smaller than t
                 add point to alsoinliers
        }
        if the number of elements in alsoinliers is > d {
            % this implies that we may have found a good model
            % now test how good it is
            bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
            thiserr = a measure of how well model fits these points
            if thiserr < besterr {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        increment iterations
    }
    return bestfit
    }}}
    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    num_samples = y_true.shape[0]
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, y_true.shape[0])
        # Y.shape is (n,m). So Y.shape[0] is n (rows)
        maybeinliers = y_true[maybe_idxs, :]
        test_points = y_true[test_idxs]
        maybemodel = y_pred.fit(maybeinliers)
        test_err = y_pred.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < threshold]  # select indices of rows with accepted points
        alsoinliers = y_true[also_idxs, :]
        if debug:
            print
            'test_err.min()', test_err.min()
            print
            'test_err.max()', test_err.max()
            print
            'numpy.mean(test_err)', np.mean(test_err)
            print
            'iteration %d:len(alsoinliers) = %d' % (
                iterations, len(alsoinliers))
        if len(alsoinliers) > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = y_pred.fit(betterdata)
            better_errs = y_pred.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
                num_inliers = np.count_nonzero(best_inlier_idxs)
                num_outliers = num_samples - num_inliers
        iterations += 1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return num_inliers, num_outliers
    else:
        return None



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

#    linear_fit, resids, rank, s = linalg.lstsq(all_data[:, input_columns],
#                                                     all_data[:, output_columns])

    # run RANSAC algorithm
    inliers, outliers = ransac(all_data, model,
                                     2, 1000, 7e3, 300,  # misc. parameters
                                     debug=debug, return_all=True)
    return inliers, outliers
print(test())

"""
    if 1:
        import pylab

        sort_idxs = numpy.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # maintain as rank-2 array

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label='RANSAC data')
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')
        pylab.plot(A_col0_sorted[:, 0],
                   numpy.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   numpy.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   numpy.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == '__main__':
    test()
"""







