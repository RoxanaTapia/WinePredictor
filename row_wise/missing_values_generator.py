import copy
import time
import random
import string
import pandas as pd


class TyposGenerator:

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

    def generate_missing_values(self, df, column_name, column_value, n):
        print("*****" * 20)
        temp = copy.deepcopy(df)
        for i in n:
            start_time = time.time()

            value = df[df[column_name].str.match(column_value)]
            value = value.shape[0]

            # increment nr of typos by 1%
            rows_affected = int((i * value) / 100.0)

            # generate typos
            aux = copy.deepcopy(temp)

            df_wine_typos = self.insert_typos(aux, column_name, column_value, rows_affected, i, value)

            # calculate metrics
            elapsed_time = time.time() - start_time
            # print("Elapsed time: {} sec".format(int(elapsed_time)))

            # yield results
            yield df_wine_typos

        print("[OK] Finished")
