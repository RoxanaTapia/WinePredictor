
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


class BasePredictor:
    def __init__(self):
        self.df_wine = self.initialize()

    @staticmethod
    def initialize():
        df_wine = pd.read_csv(
            "df_wine_horizontal_tests.csv",
            encoding='utf8',
            index_col=0
        )
        df_wine = df_wine.reset_index()

        return df_wine

    def get_null_values(self):
        print(self.df_wine.isnull().sum())

    def show_headers(self):
        print(self.df_wine.head(3))

    def get_distinct_values(self, column_name):
        """
        Retrieves distinct values in a column.
        :return:
        """
        distinct_values = pd.unique(self.df_wine[column_name])
        distinct_values = len(list(distinct_values))
        print("[OK]  Number of distinct values in column {} is {}".format(column_name, distinct_values))

    def get_value_count(self, dataframe, column_name):
        frequencies = dataframe[column_name].value_counts()
        frequencies.sort_values()
        return frequencies

    def get_selected_instances(self, frequencies):
        """
        From all different frequencies selects a high, medium and low frequency.
        :param frequencies:
        :return: a list with 3 instances (low, medium and high frequency)
        """
        result = list()

        for i in range(len(frequencies)):
            name = frequencies.index[i]
            value = frequencies[i]
            item = dict(name=name, value=value)
            result.append(item)

        # todo change for country
        max_value = result[0]
        middle_value = result[3]
        min_value = result[18]

        result = [max_value, middle_value, min_value]
        return result

    def one_hot_encoding(self, dataframe):
        df_wine = pd.get_dummies(dataframe, columns=['country'])
        df_wine = pd.get_dummies(df_wine, columns=['province'])
        df_wine = pd.get_dummies(df_wine, columns=['region'])
        df_wine = pd.get_dummies(df_wine, columns=['variety'])
        return df_wine

    def get_hyper_parameters(self, X_train, y_train):
        regressor = DecisionTreeRegressor()
        param_dist = {'max_depth': sp_randint(2, 16),
                      'min_samples_split': sp_randint(2, 16)}

        n_iter_search = 20
        clfrs = RandomizedSearchCV(
            regressor,
            param_distributions=param_dist,
            scoring='neg_mean_squared_error',
            cv=5, n_jobs=1, verbose=1,
            n_iter=n_iter_search)

        clfrs.fit(X_train, y_train)
        params = clfrs.best_params_
        return params["max_depth"], params["min_samples_split"]

    def split_data(self, dataframe):
        df_X = dataframe.drop('points', axis=1)
        df_Y = dataframe[['points']]
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.3, random_state=0)

        return X_train, X_test, y_train, y_test

    def discard_columns(self, dataframe, column_name, instance_name):
        if column_name == "region":
            df = dataframe[dataframe.region == instance_name]
        else:
            print("TBD")
            return
        return df


if __name__ == '__main__':
    base_predictor = BasePredictor()

    features = ["region"]
    modes = ["single", "all"]

    results = list()

    for feature in features:

        print("*****" * 20)
        print("Feature: ", feature)

        frequencies = base_predictor.get_value_count(base_predictor.df_wine, feature)

        # get high, medium and low frequencies
        selected_instances = base_predictor.get_selected_instances(frequencies)

        for selected in selected_instances:

            print("{} = {}".format(selected["name"], selected["value"]))

        for instance in selected_instances:
            print("Instance: ", instance)

            for mode in modes:
                df_wine_aux = deepcopy(base_predictor.df_wine)

                if mode == "single":
                    df_wine_aux = base_predictor.discard_columns(df_wine_aux, feature, instance["name"])
                    max_depth, min_samples_split = (4, 14)
                # print(df_wine_aux.shape)

                # apply one hot encoding
                df_wine_aux = base_predictor.one_hot_encoding(df_wine_aux)
                X_train, X_test, y_train, y_test = base_predictor.split_data(df_wine_aux)

                # HPO
                # max_depth, min_samples_split = base_predictor.get_hyper_parameters(X_train, y_train)
                # print(max_depth, min_samples_split)
                max_depth, min_samples_split = (9, 14)
                # Prediction
                regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
                # regressor = DecisionTreeRegressor()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)

                mae = round(metrics.mean_absolute_error(y_test, y_pred), 3)
                mse = round(metrics.mean_squared_error(y_test, y_pred), 3)
                rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 3)
                print("")
                # The evaluation metrics
                print('Mode: ', mode)
                print('Mean Absolute Error:', mae)
                print('Mean Squared Error:', mse)
                print('Root Mean Squared Error:', rmse)

                result = dict(
                    feature=feature,
                    instance=instance["name"],
                    mode=mode,
                    mae=mae,
                    mse=mse,
                    rmse=rmse,
                    delta=0,
                    precision=0,
                    recall=0
                )

                results.append(result)

            print("\n*****" * 20)

    for result in results:
        if result["mode"] == "all":
            result["mode"] = "All features"
        else:
            result["mode"] = "Single feature"

        result = "Region({instance}), {rmse}, {mode}".format(instance=result["instance"], rmse=result["rmse"], mode=result["mode"])
        print(result)

