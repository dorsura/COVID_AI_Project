import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import timedelta


class DataTransformation:
    """
    Class for performing transformations on time series data sets.
    For each transformation operator there is a inverse operator that can transform the data
    back to it's original form.
    """
    def __init__(self, data):
        """
        Init function for the DataTransformation class.
        :param data: array_like. This is the data we want to perform our transformations on. We
        assume that the data is in the format of [date, value].
        """
        self.time_series = data
        self.scaler = StandardScaler()

    def difference(self, lags):
        """
        Transform to a difference time series with a certain interval.
        :param lags: The difference operator.
        :return: Time series with applied diff.
        """
        assert lags > 0
        self.time_series = self.time_series.diff(lags).fillna(0)
        return self.time_series

    def invert_difference(self, series_with_diff, original_time_series, lags):
        """
        Invert a differenced time series according to the lag that was used to difference the data.
        The inverted time series will be saved as the DataTransaformation object's new time
        series for future operations.
        :param series_with_diff: The differenced time series.
        :param original_time_series: The original time series.
        :param lags: The difference operator.
        :return: Inverted time series.
        """
        restored = series_with_diff
        date = series_with_diff.index[0]
        end_date = series_with_diff.index[-1]
        days_to_fill = lags
        # restore the first lags according to original time series
        while days_to_fill != 0 and date <= end_date:
            if date > original_time_series.index[0]:
                restored.loc[date] = series_with_diff.loc[date] + \
                                     original_time_series.loc[date - timedelta(days=lags)].to_numpy()
            else:
                restored.loc[date] = original_time_series.loc[date].to_numpy()
            date += timedelta(days=1)
            days_to_fill -= 1
        # invert the differenced time series
        while date <= end_date:
            restored.loc[date] = series_with_diff.loc[date] + \
                                 restored.loc[date - timedelta(days=lags)]
            date += timedelta(days=1)
        self.time_series = restored
        return self.time_series

    def sqrt(self):
        """
        Transform by taking a square root of all values.
        :return: Time series with applied square root transformation.
        """
        self.time_series.iloc[:, 0] = np.sqrt(self.time_series.iloc[:, 0])
        return self.time_series

    def pow(self):
        """
        Transform by applying power of 2 to all values.
        :return: Time series with applied pow transformation.
        """
        self.time_series.iloc[:, 0] = np.power((self.time_series.iloc[:, 0]), 2)
        return self.time_series

    def log(self, increment_val=0):
        """
        Transform by applying log (i.e. ln - base e) to all values.
        :param increment_val: Log function can only be applied to numbers higher than zero,
        so if the time series has values <= 0 you should provide the increment val that will be
        added to all values in order for the log function to work properly. Note that the same
        increment_val should be provided when inverting the time series back.
        :return: Time series with applied log transformation.
        """
        self.time_series += increment_val
        self.time_series.iloc[:, 0] = np.log((self.time_series.iloc[:, 0]))
        return self.time_series

    def exp(self, decrement_val=0):
        """
        Transform by applying exponent(raise to the power of the natural exponent e) to all values.
        :param decrement_val: If this function is used to revert the log operator, a decrement
        value will be subtracted after applying the exponent function (used to restore original
        values that might have been <= 0).
        :return: Time series with applied exp transformation.
        """
        self.time_series.iloc[:, 0] = np.exp((self.time_series.iloc[:, 0]))
        self.time_series -= decrement_val
        return self.time_series

    def standardization(self):
        """
        Transform by applying standardization to the data. For each value x in our time series we
        produce a transformation value y that is given by:
        y = (x - mean) / standard_deviation
        :return: Time series with applied standardization.
        """
        values = self.time_series.values
        values = values.reshape((len(values), 1))
        # train the standardization
        self.scaler = self.scaler.fit(values)
        # print('Mean: %f, StandardDeviation: %f' % (self.scaler.mean_, math.sqrt(
        # self.scaler.var_)))

        normalized = self.scaler.transform(values)
        self.time_series.iloc[:, 0] = normalized
        return self.time_series

    def invert_standardization(self):
        """
        Apply the inverse transformation on a standardized time series.
        The inverted time series will be saved as the DataTransaformation object's new time
        series for future operations. This should be called on a standardized time series.
        :return: Original time series.
        """
        values = self.time_series.values
        values = values.reshape((len(values), 1))
        inversed = self.scaler.inverse_transform(values)
        self.time_series.iloc[:, 0] = inversed
        return self.time_series

