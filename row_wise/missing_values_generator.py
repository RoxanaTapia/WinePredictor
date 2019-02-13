import copy
import random
import pandas as pd


class MissingValuesGenerator:

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

    def insert_missing_values(self, df, column_name, column_value, n, percent, size, y_train):
        print("[OK] Inserting missing values in {} out of {} rows in {}. ({}%)".format(n, size, column_value, percent))

        # print(y_train['points'].value_counts())

        ocurrence_indexes = df.index[df[column_name] == column_value].tolist()

        # select random elements

        if len(ocurrence_indexes) < n:
            n = len(ocurrence_indexes)


        selected_indexes = random.sample(ocurrence_indexes, n)

        for index, row in df.iterrows():
            if index in selected_indexes:
                df.at[index, column_name] = "X"
                y_train.at[index, "points"] = -1

        return df, y_train, n, size, percent

    def generate_missing_values(self, df, column_name, column_value, n, y_train):
        print("*****" * 20)
        temp = copy.deepcopy(df)
        y_train_copy = copy.deepcopy(y_train)

        for i in n:
            value = df[df[column_name].str.match(column_value)]
            value = value.shape[0]
            rows_affected = int((i * value) / 100.0)
            aux = copy.deepcopy(temp)
            aux2 = copy.deepcopy(y_train_copy)
            df_wine_typos, y_train, missing_values, total, percent = self.insert_missing_values(aux, column_name, column_value, rows_affected, i, value, aux2)

            yield df_wine_typos, y_train, missing_values, total, percent

        print("[OK] Finished")
