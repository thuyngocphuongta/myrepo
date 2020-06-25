from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, fbeta_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import RANSACRegressor
from scipy import stats, linalg
import pandas as pd
import numpy as np

def dm_binary_classification_metrics(y_true, y_pred_pos_proba, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
    f1 = f1_score(y_true, y_pred)
# F2-Measure = ((1 + 2^2) * Precision * Recall) / (2^2 * Precision + Recall)
    f2 = fbeta_score(y_true, y_pred, beta=2.0)
    auc = roc_auc_score(y_true, y_pred_pos_proba)
    matthews = matthews_corrcoef(y_true, y_pred)

    row = {'Training_FK': "TEST", "Scoring_FK" : "TEST", "Input_subset_FK" : "TEST", "Client_FK" : "TEST", "Target_value" : "TEST",
           "Final_Threshold" : "TEST", "Accuracy" : accuracy, "Precision" : precision, "Recall" : recall,
           "F1-Score" : f1, "F2-Score" : f2, "Specificity" : "TEST", "AUC" : auc,
           "Matthew's_Correlation_Coefficient_(MCC)" : matthews}

    return(pd.DataFrame(row, index=[0]))



def dm_regression_metrics(y_true,y_pred_pos_proba, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared= False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
 #  adjr2 = 1-(1-r2)*(n-1)/(n-p-1)
 #  where n is number of observations in sample and p is number of independent variables in model
    def mean_squared_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        return np.mean(((y_true - y_pred) / y_true) ** 2) * 100

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mspe = mean_squared_percentage_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    spearman = stats.spearmanr(y_true, y_pred)
    pearson = stats.pearsonr(y_true, y_pred)

    row = {"Training_FK": "TEST", "Scoring_FK": "TEST", "Input_subset_FK": "TEST", "Client_FK": "TEST",
           "MSE": mse, "RMSE": rmse, "MAE": mae,
           "R_squared": r2, "Adjusted_R_squared": "TEST",
           "MSPE": mspe, "MAPE": mape,
           "RMSLE": rmsle,
           "Spearman": spearman, "Pearson": pearson}

    return(pd.DataFrame(row, index=[0]))

def dm_regression_inlier_ratio(y_true,y_pred_pos_proba, y_pred):

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

    threshold = np.linspace(0, 1, num=100)
    inliers, outliers = ransac(y_true, y_pred, 2,1000,threshold=threshold,d=0.8*y_true,debug=False, return_all=True )

    row = {"Training_FK": "TEST", "Scoring_FK": "TEST", "Input_subset_FK": "TEST", "Client_FK": "TEST",
           "Threshold": "TEST", "Number_of_Inliers": inliers, "Number_of_Outliers": outliers
           }

    return(pd.DataFrame(row, index=[0]))

def dm_multi_classification_metrics(y_true, y_pred_pos_proba, y_pred):
    avg_accuracy = balanced_accuracy_score(y_true, y_pred) # is it recall macro?
    error = 1-accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f2_micro = fbeta_score(y_true, y_pred, beta=2.0, average='micro')
    f2_macro = fbeta_score(y_true, y_pred, beta=2.0, average='macro')

    row = {"Training_FK": "TEST", "Scoring_FK" : "TEST", "Input_subset_FK" : "TEST", "Client_FK" : "TEST",
           "Average_Accuracy": avg_accuracy, "Error_rate": error, "Precision_micro": precision_micro, "Recall_micro": recall_micro,
           "F1-Score_micro": f1_micro, "F2-Score_micro" : f2_micro,
           "Precision_macro": precision_macro, "Recall_macro": recall_macro, "F1-Score_macro": f1_macro, "F2-Score_macro" : f2_macro}

    return(pd.DataFrame(row, index=[0]))