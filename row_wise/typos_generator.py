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

    def randomize(self, value, n):
        # makes harder to match the original value replacing n of the original chars
        letters = string.ascii_letters
        indexes = range(0, len(value))
        indexes = random.sample(indexes, n)
        for index in indexes:
            temp = list(value)
            temp[index] = random.choice(letters)
            value = ''.join(temp)
        return value

    def generate_random_typo(self, value):
        letters = string.ascii_letters
        degree = random.randint(1, len(value))  # add at least 1 typo
        indexes = range(0, len(value))
        indexes = random.sample(indexes, degree)

        for index in indexes:
            temp = list(value)
            temp[index] = temp[index] + random.choice(letters)
            value = ''.join(temp)

        return self.randomize(value, 1)

    def insert_typos(self, df, column_name, column_value, n, percent, size):
        print("[OK] Inserting typos in {} out of {} rows in {}. ({}%)".format(n, size, column_value, percent))

        ocurrence_indexes = df.index[df[column_name] == column_value].tolist()

        # select random elements
        selected_indexes = random.sample(ocurrence_indexes, n)
        for index, row in df.iterrows():
            if index in selected_indexes:
                value = self.generate_random_typo(column_value)
                # print("[OK] Generated typo...", value)
                df.at[index, column_name] = value
        return df

    def generate_dirty_data(self, df, column_name, column_value, n):
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
