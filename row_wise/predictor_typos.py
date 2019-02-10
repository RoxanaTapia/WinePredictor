import copy
import pandas as pd
import numpy as np
import time
from row_wise import typos_generator
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


class PredictorTypos:

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

    def discard_columns(self, dataframe, column_name, instance_name):
        if column_name == "region":
            df = dataframe[dataframe.region == instance_name]
        else:
            print("TBD")
            return
        return df

    def split_data(self, dataframe):
        df_X = dataframe.drop('points', axis=1)
        df_Y = dataframe[['points']]
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.3, random_state=0)

        return X_train, X_test, y_train, y_test

    def one_hot_encoding(self, dataframe):
        lb = LabelBinarizer()
        # dataframe = pd.DataFrame(lb.fit_transform(dataframe['country']), columns=lb.classes_)
        # dataframe = pd.DataFrame(lb.fit_transform(dataframe['province']), columns=lb.classes_)
        # dataframe = pd.DataFrame(lb.fit_transform(dataframe['region']), columns=lb.classes_)
        # dataframe = pd.DataFrame(lb.fit_transform(dataframe['variety']), columns=lb.classes_)
        df_wine = pd.get_dummies(dataframe, columns=['country'])
        df_wine = pd.get_dummies(df_wine, columns=['province'])
        df_wine = pd.get_dummies(df_wine, columns=['region'])
        df_wine = pd.get_dummies(df_wine, columns=['variety'])
        return df_wine


if __name__ == '__main__':

    predictor_typos = PredictorTypos()
    typos_generator = typos_generator.TyposGenerator()

    features = ["region"]
    modes = ["single", "all"]
    approaches = ["training", "test"]
    error_percentages = [25, 50, 75]

    for feature in features:
        print("*****" * 20)
        print("Feature: ", feature)

        frequencies = predictor_typos.get_value_count(predictor_typos.df_wine, feature)
        selected_instances = predictor_typos.get_selected_instances(frequencies)

        for instance in selected_instances:
            print("Instance: ", instance)

            for mode in modes:
                df_wine_aux = copy.deepcopy(predictor_typos.df_wine)

                if mode == "single":
                    df_wine_aux = predictor_typos.discard_columns(df_wine_aux, feature, instance["name"])

                X_train, X_test, y_train, y_test = predictor_typos.split_data(df_wine_aux)

                X_test_copy = copy.deepcopy(X_test)
                X_train_copy = copy.deepcopy(X_train)

                # typos application
                for approach in approaches:

                    if approach == "test":
                        # apply typos to x test
                        # do one hot encoding of constant
                        X_train = predictor_typos.one_hot_encoding(X_train)

                        for X_test_typos in typos_generator.generate_dirty_data(X_test_copy, feature, instance["name"], error_percentages):

                            X_test_typos = predictor_typos.one_hot_encoding(X_test_typos)
                            X_test_typos, X_train = X_test_typos.align(X_train, join='outer', axis=1, fill_value=0)

                            regressor = DecisionTreeRegressor()
                            regressor.fit(X_train, y_train)
                            y_pred = regressor.predict(X_test_typos)

                            mae = round(metrics.mean_absolute_error(y_test, y_pred), 4)
                            mse = round(metrics.mean_squared_error(y_test, y_pred), 4)
                            rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
                            print("")
                            # The evaluation metrics
                            print('Mode: ', mode)
                            print('Approach: ', approach)
                            print('Mean Absolute Error:', mae)
                            print('Mean Squared Error:', mse)
                            print('Root Mean Squared Error:', rmse)
                    else:
                        # apply typos to x train
                        X_test = predictor_typos.one_hot_encoding(X_test)

                        for X_train_typos in typos_generator.generate_dirty_data(X_train_copy, feature, instance["name"], error_percentages):

                            X_train_typos = predictor_typos.one_hot_encoding(X_train_typos)
                            X_train_typos, X_test = X_train_typos.align(X_test, join='outer', axis=1, fill_value=0)

                            regressor = DecisionTreeRegressor()
                            regressor.fit(X_train_typos, y_train)
                            y_pred = regressor.predict(X_test)

                            mae = round(metrics.mean_absolute_error(y_test, y_pred), 4)
                            mse = round(metrics.mean_squared_error(y_test, y_pred), 4)
                            rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
                            print("")
                            # The evaluation metrics
                            print('Mode: ', mode)
                            print('Approach: ', approach)
                            print('Mean Absolute Error:', mae)
                            print('Mean Squared Error:', mse)
                            print('Root Mean Squared Error:', rmse)
