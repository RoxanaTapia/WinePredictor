import copy
import pandas as pd
import numpy as np
from row_wise import missing_values_generator
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


class PredictorMissingValues:

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
        df_wine = pd.get_dummies(dataframe, columns=['country'])
        df_wine = pd.get_dummies(df_wine, columns=['province'])
        df_wine = pd.get_dummies(df_wine, columns=['region'])
        df_wine = pd.get_dummies(df_wine, columns=['variety'])
        return df_wine

    def detect_missing_values(self, df, column_name, column_value, expected_distinct_values, correctness, y_train):
        df = df.dropna(subset=[column_name])
        df = df[df.region != "X"]
        y_train = y_train[y_train.points != -1]
        print("[OK] Clean correctness: {}%".format(correctness))
        return df, y_train


if __name__ == '__main__':

    general_results = list()

    predictor_missing_values = PredictorMissingValues()
    missing_values_generator = missing_values_generator.MissingValuesGenerator()

    features = ["region"]
    modes = ["single", "all"]
    approaches = ["training", "test"]
    error_percentages = [25, 50, 75]

    experiments = list()

    for feature in features:
        print("*****" * 20)
        print("Feature: ", feature)

        frequencies = predictor_missing_values.get_value_count(predictor_missing_values.df_wine, feature)
        selected_instances = predictor_missing_values.get_selected_instances(frequencies)

        for instance in selected_instances:
            print("Instance: ", instance)

            for mode in modes:
                df_wine_aux = copy.deepcopy(predictor_missing_values.df_wine)

                if mode == "single":
                    df_wine_aux = predictor_missing_values.discard_columns(df_wine_aux, feature, instance["name"])

                X_train, X_test, y_train, y_test = predictor_missing_values.split_data(df_wine_aux)

                X_test_copy = copy.deepcopy(X_test)
                X_train_copy = copy.deepcopy(X_train)

                y_train_copy = copy.deepcopy(y_train)

                # typos application
                for approach in approaches:

                    if approach == "test":
                        # apply typos to x test
                        # do one hot encoding of constant
                        X_train_aux = copy.deepcopy(predictor_missing_values.one_hot_encoding(X_train))

                        for X_test_typos, y_train_typos in missing_values_generator.generate_missing_values(X_test_copy, feature, instance["name"], error_percentages, y_train_copy):

                            experiment = dict(mode=mode, approach=approach, constant=X_train_aux, variant=X_test_typos, feature=feature, instance=instance["name"], y_train=y_train_typos, y_test=y_test)
                            experiments.append(experiment)

                            X_test_typos = predictor_missing_values.one_hot_encoding(X_test_typos)
                            X_test_typos, X_train_aux = X_test_typos.align(X_train_aux, join='outer', axis=1, fill_value=0)

                            regressor = DecisionTreeRegressor()
                            regressor.fit(X_train_aux, y_train)
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

                            result = dict(
                                feature=feature,
                                instance=instance["name"],
                                mode=mode,
                                approach=approach,
                                mae=mae,
                                mse=mse,
                                rmse=rmse,
                                delta=0,
                                precision=0,
                                recall=0
                            )
                            general_results.append(result)
                    else:
                        # apply typos to x train
                        X_test_aux = copy.deepcopy(predictor_missing_values.one_hot_encoding(X_test))
                        y_train_copy_aux = copy.deepcopy(y_train_copy)

                        for X_train_typos, y_train_typos in missing_values_generator.generate_missing_values(X_train_copy, feature, instance["name"], error_percentages, y_train_copy):

                            experiment = dict(mode=mode, approach=approach, constant=X_test_aux, variant=X_train_typos, feature=feature, instance=instance["name"], y_train=y_train_typos, y_test=y_test)
                            experiments.append(experiment)
                            X_train_typos = predictor_missing_values.one_hot_encoding(X_train_typos)
                            X_train_typos, X_test_aux = X_train_typos.align(X_test_aux, join='outer', axis=1, fill_value=0)

                            regressor = DecisionTreeRegressor()
                            regressor.fit(X_train_typos, y_train)
                            y_pred = regressor.predict(X_test_aux)

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

                            result = dict(
                                feature=feature,
                                instance=instance["name"],
                                mode=mode,
                                approach=approach,
                                mae=mae,
                                mse=mse,
                                rmse=rmse,
                                delta=0,
                                precision=0,
                                recall=0
                            )
                            general_results.append(result)
    # Claening

    expected_distinct_values = ['Central Coast', 'Napa', 'Sicilia']
    correctness = 100

    # print("CLEANED")
    #
    # for experiment in experiments:
    #
    #     if experiment["approach"] == "test":
    #         X_test_typos, y_train = predictor_missing_values.detect_missing_values(experiment["variant"], experiment["feature"], experiment["instance"], expected_distinct_values, correctness, y_train)
    #
    #         X_test_typos = predictor_missing_values.one_hot_encoding(X_test_typos)
    #         X_test_typos, X_train = X_test_typos.align(experiment["constant"], join='outer', axis=1, fill_value=0)
    #
    #         regressor = DecisionTreeRegressor()
    #         regressor.fit(X_train, experiment["y_train"])
    #         y_pred = regressor.predict(X_test_typos)
    #
    #         y_test = experiment["y_test"]
    #
    #         # y_test, y_pred = y_test.align(y_pred, join='outer', axis=1, fill_value=0)
    #
    #         mae = round(metrics.mean_absolute_error(y_test, y_pred), 4)
    #         mse = round(metrics.mean_squared_error(y_test, y_pred), 4)
    #         rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
    #
    #         # The evaluation metrics
    #         print('Mode: ', experiment["mode"])
    #         print('Approach: ', experiment["approach"])
    #         print('Feature: ', experiment["feature"])
    #         print('Instance: ', experiment["instance"])
    #         print('Mean Absolute Error:', mae)
    #         print('Mean Squared Error:', mse)
    #         print('Root Mean Squared Error:', rmse)
    #
    #         result = dict(
    #             feature=experiment["feature"],
    #             instance=experiment["instance"],
    #             mode=experiment["mode"],
    #             approach=experiment["approach"],
    #             mae=mae,
    #             mse=mse,
    #             rmse=rmse,
    #             precision=0,
    #             recall=0
    #         )
    #         general_results.append(result)
    #     else:
    #         X_train_typos = experiment["variant"]
    #
    #         X_train_typos, y_train = predictor_missing_values.detect_missing_values(X_train_typos, experiment["feature"], experiment["instance"], expected_distinct_values, correctness, experiment["y_train"])
    #
    #         X_train_typos = predictor_missing_values.one_hot_encoding(X_train_typos)
    #         X_train_typos, X_test = X_train_typos.align(experiment["constant"], join='outer', axis=1, fill_value=0)
    #
    #         regressor = DecisionTreeRegressor()
    #         regressor.fit(X_train_typos, y_train)
    #         y_pred = regressor.predict(X_test)
    #
    #         y_test = experiment["y_test"]
    #
    #         mae = round(metrics.mean_absolute_error(y_test, y_pred), 4)
    #         mse = round(metrics.mean_squared_error(y_test, y_pred), 4)
    #         rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
    #         print("")
    #         # The evaluation metrics
    #         print('Mode: ', experiment["mode"])
    #         print('Approach: ', experiment["approach"])
    #         print('Feature: ', experiment["feature"])
    #         print('Instance: ', experiment["instance"])
    #         print('Mean Absolute Error:', mae)
    #         print('Mean Squared Error:', mse)
    #         print('Root Mean Squared Error:', rmse)
    #
    #         result = dict(
    #             feature=experiment["feature"],
    #             instance=experiment["instance"],
    #             mode=experiment["mode"],
    #             approach=experiment["approach"],
    #             mae=mae,
    #             mse=mse,
    #             rmse=rmse,
    #             precision=0,
    #             recall=0
    #         )
    #         general_results.append(result)
