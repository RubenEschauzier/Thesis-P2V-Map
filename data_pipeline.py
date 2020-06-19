import datetime
from collections import defaultdict
from itertools import permutations
import pandas as pd
import numpy as np
import json

from helper_functions import get_filepath
from skipgram_model import centerContextLabel, P2VTensorFlow


class ExcelReader:
    def __init__(self, num_files):
        self.num_files = num_files

    def set_path(self, folder_location, filename, num_files=1):
        # This creates filepaths based on the assumption that all files that need to be read
        # use filename1, filename2, filename3
        if num_files > 1:
            path = [get_filepath(folder_location, filename + str(i) + ".xlsx") for i in range(1, num_files + 1)]
        elif num_files == 1:
            path = get_filepath(folder_location, filename + ".xlsx")
        else:
            print("please enter > 0 files te read")
        return path

    def read_split_excel(self, filename, folder_location):
        paths = self.set_path(folder_location, filename, self.num_files)
        dfs_to_concat = []

        for i, path in enumerate(paths):
            dfs_to_concat.append(pd.read_excel(path, index_col=0))
            self.INFO_reading_file(len(paths), i + 1)

        return pd.concat(dfs_to_concat)

    def read_single_excel(self, file_location, folder_location):
        paths = self.set_path(folder_location, file_location, 1)
        data = pd.read_excel(paths)
        self.INFO_reading_file(1, 1)
        return data

    @staticmethod
    def INFO_reading_file(total_files, current_file):
        time = datetime.datetime.now().strftime("%H:%M:%S")
        print("{} INFO: Read {} out of {} files into Dataframe".format(time, current_file, total_files))


class InstaCartPipeline(ExcelReader):
    def __init__(self, threshold, file_name, file_dir, basket_col_name, product_col_name, data=None, num_excel_files=1):
        super(InstaCartPipeline, self).__init__(num_excel_files)
        self.threshold = threshold
        if data is None and num_excel_files > 1:
            self.data = self.read_split_excel(file_name, file_dir)
        if data is None and num_excel_files == 1:
            self.data = self.read_single_excel(file_name, file_dir)
        if data is not None:
            self.data = data
        self.filtered = False
        self.baskets = self.create_baskets(basket_col_name, product_col_name)
        self.num_products = self.create_product_count(product_col_name)
        self.frequencies = self.create_frequencies()
        self.data = None

    def create_baskets(self, basket_id_name=None, product_id_name=None):
        if basket_id_name and product_id_name is None:
            baskets = defaultdict(list)
            for index, row in self.data.iterrows():
                baskets[row['order_id']].append(row['product_id'])
            return baskets
        else:
            baskets = defaultdict(list)
            for index, row in self.data.iterrows():
                baskets[row[basket_id_name]].append(row[product_id_name])
            return baskets

    def create_product_count(self, product_id_name):
        max_product = self.data[product_id_name].max()
        print(max_product)
        return max_product

    def create_frequencies(self):
        # Create frequency dictionary
        frequencies = {}
        for basket, products in self.baskets.items():
            # List that enables us to not count products twice if they occur twice in the same basket
            already_counted = []
            for product in products:
                if product not in frequencies and product not in already_counted:
                    frequencies[product] = 1
                elif product not in already_counted:
                    frequencies[product] = frequencies[product] + 1
                already_counted.append(product)

        # Make all not sold products 0 in frequency dict
        for product1 in range(self.num_products):
            if product1 not in frequencies.keys():
                frequencies[product1] = 0
        return frequencies

    def filter_rare_observations(self):
        # Filter products with frequencies < threshold
        for basket, products in self.baskets.items():
            products[:] = (x for x in products if self.frequencies[x] > self.threshold)
            self.baskets[basket] = products
        self.filtered = True

    def create_center_contex_pairs(self):
        neg_sample_generator = NegativeSampleGenerator(True, self.frequencies, .75,
                                                       2 ** 31 - 1, 20)
        center_product_cache = []
        pos_context_product_cache = []
        neg_context_product_cache = []
        print("Creating context")
        if self.filtered:
            for i, (basket, products) in enumerate(self.baskets.items()):

                # Get all permutations and convert into a list
                center_context = permutations(products, 2)
                center_context_pairs = list(map(list, list(center_context)))

                # Get negative samples for the basket
                neg_samples = neg_sample_generator.draw_negative_sample(np.array(center_context_pairs))

                # Add negative sample to permutation where it belongs
                for x, cen_cont in enumerate(center_context_pairs):
                    center_product_cache.append(cen_cont[0])
                    pos_context_product_cache.append(cen_cont[1])
                    neg_context_product_cache.append(neg_samples[x])

                if i % 100000 == 0:
                    print('Saving to file')
                    with open(r"large_data\center_products_simulated_test", 'ab') as f:
                        np.savetxt(f, np.array(center_product_cache),
                                   delimiter=',',
                                   fmt='%d')
                    with open(r"large_data\positive_context_products_simulated_test", 'ab') as file:
                        np.savetxt(file, np.array(pos_context_product_cache),
                                   delimiter=',',
                                   fmt='%d')
                    with open(r"large_data\negative_context_products_simulated_test", 'ab') as fi:
                        np.savetxt(fi, np.array(neg_context_product_cache),
                                   delimiter=',',
                                   fmt='%d')
                    center_product_cache = []
                    pos_context_product_cache = []
                    neg_context_product_cache = []

            with open(r"large_data\center_products_simulated_test", 'ab') as f:
                np.savetxt(f, np.array(center_product_cache),
                           delimiter=',',
                           fmt='%d')
            with open(r"large_data\positive_context_products_simulated_test", 'ab') as file:
                np.savetxt(fname=file, X=np.array(pos_context_product_cache),
                           delimiter=',',
                           fmt='%d')
            with open(r"large_data\negative_context_products_simulated_test", 'ab') as fi:
                np.savetxt(fi, np.array(neg_context_product_cache),
                           delimiter=',',
                           fmt='%d')
        else:
            time = datetime.datetime.now().strftime("%H:%M:%S")
            print("{} WARNING: First filter out infrequent products".format(time))

        return np.array(center_product_cache, dtype=np.int32), np.array(pos_context_product_cache, dtype=np.int32), \
               np.array(neg_context_product_cache, dtype=np.int32)

    def get_baskets(self):
        return self.baskets

    def get_num_products(self):
        return self.num_products

    def get_frequencies(self):
        return self.frequencies


class NegativeSampleGenerator:

    def __init__(self, suppress, frequency, pow, range_int, n_neg_samples):
        self.n_neg_samples = n_neg_samples
        self.range = range_int
        self.pow = pow
        self.suppress_collision = suppress
        self.frequency = frequency
        self.products = np.array(list(self.frequency.keys()))
        self.count_table = self.build_cumulative()

    def draw_negative_sample(self, pos_context):
        neg_samples = np.empty((pos_context.shape[0], self.n_neg_samples))
        neg_samples.fill(np.nan)
        stop = False

        while not stop:
            sample_index = np.isnan(neg_samples)
            n_draws = np.sum(sample_index)
            draws = np.random.randint(0, self.range, n_draws)
            negative_sample_index = np.searchsorted(self.count_table, draws)

            neg_samples_new = self.products[negative_sample_index]
            neg_samples[sample_index] = neg_samples_new

            for neg_sample in neg_samples:
                if neg_sample in pos_context:
                    i = list(neg_samples).index(neg_sample)
                    neg_samples[i] = np.nan

            if np.all(neg_samples != np.nan):
                stop = True

        return neg_samples.astype(int)

    def build_cumulative(self):
        cdf = np.array(list(self.frequency.values()), dtype=float) ** self.pow
        cum_count = np.cumsum(cdf / sum(cdf))
        count_table = ((cum_count * self.range).round())
        return count_table


if __name__ == '__main__':
    pipeline = InstaCartPipeline(25, "simulated_data", "large_data", "basket_id", "j", num_excel_files=4)
    pipeline.filter_rare_observations()
    center, pos_context, neg_context = pipeline.create_center_contex_pairs()

