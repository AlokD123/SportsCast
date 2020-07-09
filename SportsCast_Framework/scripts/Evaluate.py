import logging
import numpy as np
import pandas as pd
import itertools

class Evaluator:
    ''' Provides methods for evaluation '''
    @classmethod
    def calculate_errors(cls,residuals):
        '''
        Calculates errors based on residuals

        Parameters
        ----
        residuals: vector

        Returns
        ---
        mfe: mean forecasting error
        
        mae: mean abs error

        rmse: root MSE

        '''
        num_residuals = len(residuals)
        mfe = (residuals.sum() / num_residuals).tolist()[0]
        mae = (residuals.abs().sum() / num_residuals).tolist()[0]
        rmse = (residuals.pow(2).sum().pow(0.5)).tolist()[0]
        residuals = residuals.values
        residuals = [value.item() for value in residuals]
        return mfe, mae, rmse

    @classmethod
    def calculate_test_residuals(cls,prediction_array, test_data):
        '''
        Calculates test residuals based on prediction and test data

        Parameters
        ----
        prediction_array: prediction

        test_data: labels

        Returns
        ---
        residuals: vector of residuals

        '''
        prediction_array = prediction_array.reshape(len(test_data), 1)
        test_data = test_data.values
        residuals = np.subtract(test_data, prediction_array)
        residuals = pd.residuals.tolist()
        residuals = pd.DataFrame(residuals)
        return residuals

    @classmethod
    def diffN(cls,arr1,N):
        '''
        Calculates N-step differences for an array (to calculate modified-MASE)

        Parameters
        ----
        arr1: array (vector)

        N: number of steps

        Returns
        ---
        diff_arr: N-step differenced array

        '''
        assert N>0, "Invalid difference!"
        f = lambda arr,i,n : 0 if (i+n+1)>len(arr) else arr[i+n]-arr[i]
        diff_arr = np.zeros(len(arr1))
        for j in range(len(arr1)):
            diff_arr[j] = f(arr1,j,N)
        return diff_arr
    
    @classmethod
    def calculate_errors_trackingSig(cls,target_data, prediction_array=None, residuals=None, mase_forecast=1):
        '''
        Calculates all metrics

        Parameters
        ----
        target_data: labels
        
        prediction_array: predictions

        residuals: vector

        mase_forecast: number of steps for forecasting used

        Returns
        ---
        Previously defined metrics and...

        tracking: tracking signal metric

        mase: a MODIFIED version of the mean abs scaled error based on N-steps forecasting (defined in mase_forecast)

        '''

        try:
            if residuals is None and prediction_array is not None:
                residuals = Evaluator.calculate_test_residuals(prediction_array,target_data)
            elif prediction_array is None and residuals is not None:
                pass
            elif prediction_array is None and residuals is None:
                raise ValueError("Missing both prediction and residuals!")
            else:
                raise ValueError("Don't specify both prediction and residuals!")

            assert len(residuals)>0, "Missing residuals"
            assert len(target_data)>1, "Missing targets or only one point"

            mfe, mae, rmse = Evaluator.calculate_errors(residuals)
            tracking = sum(residuals.values)[0]/mae
            assert np.isscalar(sum(residuals.values)[0]), f"Error with sum for tracking signal. Sum = {sum(residuals.values)}"
            assert np.isscalar(tracking), f"Error with tracking signal. Residuals={residuals}"

            targets = np.array([val[0] for val in target_data.values])
            assert len(targets.shape)==1, f"Error with target dims. Shape={targets.shape}"
            first_diff_targets = Evaluator.diffN(targets, mase_forecast)
            assert len(first_diff_targets)>0, f"First diff is absent. First diffs={first_diff_targets}"
            oneNorm_first_diff_targets = sum(np.absolute(first_diff_targets))
            if not np.isscalar(oneNorm_first_diff_targets):
                logging.error(f'\nTargets: {targets}')
                logging.error(f'\nFirst diffs targets: {first_diff_targets}')
                logging.error(f'\nAbsolute value: {np.absolute(first_diff_targets)}')
                logging.error(f'\nSum: {oneNorm_first_diff_targets}')
                raise ValueError("Error with sum for mase")
            mean_first_diff_targets = oneNorm_first_diff_targets/len(first_diff_targets)
            mase = mae/mean_first_diff_targets
        except AssertionError as err:
            print(f"Error in calculating resids/tracking: {err}")
            return None 
        except ValueError as err:
            print(f"Error in calculating resids/tracking: {err}")
            return None
        return mfe, mae, rmse, mase, tracking, residuals

    @classmethod
    def batch_array(cls,iterable, n=1):
        '''
        Batches a list into parts, with the possibility of remainder

        Parameters
        ----
        n: length of batch

        Returns
        ----
        iterator to get batch
        '''

        totLen = len(iterable)
        for ndx in range(0, totLen, n):
            yield iterable[ndx:min(ndx + n, totLen)]

    @classmethod
    def batch_generator(cls,iterable, n):
        '''
        Batches a generic iterable into parts, with the possibility of remainder

        Parameters
        ----
        n: length of batch

        Returns
        ----
        chunk: iterator to get batch
        '''
        
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk