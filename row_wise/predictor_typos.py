import copy
import pandas as pd
import numpy as np
from row_wise import typos_generator
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
        df_wine = pd.get_dummies(dataframe, columns=['country'])
        df_wine = pd.get_dummies(df_wine, columns=['province'])
        df_wine = pd.get_dummies(df_wine, columns=['region'])
        df_wine = pd.get_dummies(df_wine, columns=['variety'])
        return df_wine

    def clean_value(self, value, original, percent):
        typos_chars = list(value)
        items = list(original)

        check_list = dict()
        for item in items:
            check_list[item] = False

        # check if value contains all chars of original
        for typo in typos_chars:
            if typo in original:
                if check_list[typo]:
                    continue
                check_list[typo] = True

        # how many characters from the original were found,
        # correctnes 100 means, all characters in the original were found
        correctness = int((percent * len(original)) / 100.0)
        corrected = len([x for x in check_list.values() if x])

        if corrected < correctness:
            # print("[ERROR] Couldnt clean: {}.".format(value))
            return value

        return original

    def detect_typos(self, df, column_name, column_value, expected_distinct_values, correctness):
        i = 0
        b = 0
        print("[OK] Clean correctness: {}%".format(correctness))
        for index, row in df.iterrows():
            value = row[column_name]
            if value in expected_distinct_values:
                continue

            new_value = self.clean_value(value, column_value, correctness)

            if value != new_value:
                df.at[index, column_name] = new_value
                # print("[OK] Cleaning {} results in {}".format(value, new_value))
                i += 1
            else:
                # discard the row that couldnt be cleaned
                # df.at[index, column_name] = np.NaN
                b += 1

        # print("[OK] Typos detected: {}".format(b+i))
        print("[OK] Cleaned: {} rows".format(i))
        # print("[WARN] Couldn't clean {} rows".format(b))
        return df, i


if __name__ == '__main__':

    predictor_typos = PredictorTypos()
    typos_generator = typos_generator.TyposGenerator()

    features = ["region"]
    modes = ["all", "single"]
    approaches = ["test"]
    error_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    results = list()

    experiments = list()

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
                    max_depth, min_samples_split = (4, 14)

                max_depth, min_samples_split = (9, 14)
                X_train, X_test, y_train, y_test = predictor_typos.split_data(df_wine_aux)

                X_test_copy = copy.deepcopy(X_test)
                X_train_copy = copy.deepcopy(X_train)

                # typos application
                for approach in approaches:

                    if approach == "test":
                        # apply typos to x test
                        # do one hot encoding of constant
                        X_train_aux = copy.deepcopy(predictor_typos.one_hot_encoding(X_train))

                        for X_test_typos, errors, total, percent in typos_generator.generate_dirty_data(X_test_copy, feature, instance["name"], error_percentages):

                            experiment = dict(mode=mode, approach=approach, constant=X_train_aux, variant=X_test_typos, feature=feature, instance=instance["name"], y_train=y_train, y_test=y_test, max_depth=max_depth, min_samples_split=min_samples_split, percent=percent)
                            experiments.append(experiment)

                            X_test_typos = predictor_typos.one_hot_encoding(X_test_typos)
                            X_test_typos, X_train_aux = X_test_typos.align(X_train_aux, join='outer', axis=1, fill_value=0)

                            regressor = DecisionTreeRegressor()
                            regressor.fit(X_train_aux, y_train)
                            y_pred = regressor.predict(X_test_typos)

                            mae = round(metrics.mean_absolute_error(y_test, y_pred), 3)
                            mse = round(metrics.mean_squared_error(y_test, y_pred), 3)
                            rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 3)
                            print("")
                            # The evaluation metrics
                            print('Mode: ', mode)
                            print('Approach: ', approach)
                            print('Mean Absolute Error:', mae)
                            print('Mean Squared Error:', mse)
                            print('Root Mean Squared Error:', rmse)

                            result = dict(feature=feature, instance=instance["name"], mode=mode, approach=approach, rmse=rmse, errors=errors, total=total, percentage=percent)
                            results.append(result)

                    else:
                        # apply typos to x train
                        X_test_aux = copy.deepcopy(predictor_typos.one_hot_encoding(X_test))

                        for X_train_typos in typos_generator.generate_dirty_data(X_train_copy, feature, instance["name"], error_percentages):

                            experiment = dict(mode=mode, approach=approach, constant=X_test_aux, variant=X_train_copy, feature=feature, instance=instance["name"], y_train=y_train, y_test=y_test, max_depth=max_depth, min_samples_split=min_samples_split)
                            experiments.append(experiment)

                            X_train_typos = predictor_typos.one_hot_encoding(X_train_typos)
                            X_train_typos, X_test_aux = X_train_typos.align(X_test_aux, join='outer', axis=1, fill_value=0)

                            regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
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

    print("=====" * 20)
    print("RESULTS DIRTY")
    print("=====" * 20)
    for result in results:
        if result["mode"] == "all":
            result["mode"] = "All features"
        else:
            result["mode"] = "Single feature"

        r = "Region({instance}), {percentage}%, {rmse}, {mode}, {errors}, {total}".format(instance=result["instance"], percentage=result["percentage"], rmse=result["rmse"], mode=result["mode"], errors=result["errors"], total=result["total"])
        print(r)
    print("=====" * 20)

    # cleaning

    expected_distinct_values = ['Central Coast', 'Napa', 'Sicilia']
    correctness = 80

    result_cleaned = list()

    for experiment in experiments:

        if experiment["approach"] == "test":
            X_test_typos, cleaned = predictor_typos.detect_typos(experiment["variant"], experiment["feature"], experiment["instance"], expected_distinct_values, correctness)

            X_test_typos = predictor_typos.one_hot_encoding(X_test_typos)
            X_test_typos, X_train = X_test_typos.align(experiment["constant"], join='outer', axis=1, fill_value=0)

            regressor = DecisionTreeRegressor()
            regressor.fit(X_train, experiment["y_train"])
            y_pred = regressor.predict(X_test_typos)

            mae = round(metrics.mean_absolute_error(experiment["y_test"], y_pred), 4)
            mse = round(metrics.mean_squared_error(experiment["y_test"], y_pred), 4)
            rmse = round(np.sqrt(metrics.mean_squared_error(experiment["y_test"], y_pred)), 4)

            # The evaluation metrics
            print('Mode: ', experiment["mode"])
            print('Approach: ', experiment["approach"])
            print('Feature: ', experiment["feature"])
            print('Instance: ', experiment["instance"])
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('Root Mean Squared Error:', rmse)

            result = dict(feature=experiment["feature"], instance=experiment["instance"], mode=experiment["mode"], approach=experiment["approach"], rmse=rmse,
                          cleaned=cleaned, precision=correctness, percent=experiment["percent"])

            result_cleaned.append(result)
        else:

            X_train_typos = predictor_typos.detect_typos(experiment["variant"], experiment["feature"], experiment["instance"], expected_distinct_values, correctness)
            X_train_typos = predictor_typos.one_hot_encoding(X_train_typos)
            X_train_typos, X_test = X_train_typos.align(experiment["constant"], join='outer', axis=1, fill_value=0)

            regressor = DecisionTreeRegressor(max_depth=experiment["max_depth"], min_samples_split=experiment["min_samples_split"])
            regressor.fit(X_train_typos, experiment["y_train"])
            y_pred = regressor.predict(X_test)

            mae = round(metrics.mean_absolute_error(experiment["y_test"], y_pred), 4)
            mse = round(metrics.mean_squared_error(experiment["y_test"], y_pred), 4)
            rmse = round(np.sqrt(metrics.mean_squared_error(experiment["y_test"], y_pred)), 4)
            print("")
            # The evaluation metrics
            print('Mode: ', experiment["mode"])
            print('Approach: ', experiment["approach"])
            print('Feature: ', experiment["feature"])
            print('Instance: ', experiment["instance"])
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('Root Mean Squared Error:', rmse)

    print("=====" * 20)
    print("RESULTS CLEANED")
    print("====="*20)
    for result in result_cleaned:
        if result["mode"] == "all":
            result["mode"] = "All features"
        else:
            result["mode"] = "Single feature"
        r = "Region({instance}), {percentage}%, {rmse}, {mode}, {cleaned}, {precision}".format(instance=result["instance"], percentage=result["percent"], rmse=result["rmse"], mode=result["mode"], cleaned=result["cleaned"], precision=result["precision"])
        print(r)
    print("=====" * 20)
