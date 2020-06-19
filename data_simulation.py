import os
import sys
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
from helper_functions import get_filepath, create_html, INFO_LOGGING
import warnings


class basket_simulator (INFO_LOGGING):

    def __init__(self, C, J_c, T, I, num_consumers):
        super(basket_simulator, self).__init__()
        self.num_consumers = num_consumers
        self.C = C
        self.J_c = J_c
        self.T = T
        self.I = I
        self.correlation_matrix = self.create_correlation_matrix()
        self.consumers = self.create_consumers()

    def create_consumers(self):
        self.INFO_creating_consumers()

        consumers = []
        for i in tqdm(range(self.num_consumers), file = sys.stdout):
            cons = Consumer(20, -.5, 2, 15, 2, 1, i)
            consumers.append(cons)
        return consumers

    def draw_multivariate_normal(self):
        mean = np.zeros(20)
        return np.random.multivariate_normal(mean, self.correlation_matrix)

    def simulate_consumer(self):
        dict_data = defaultdict(list)
        basket_id = 0
        for week in range(self.T):
            self.INFO_simulation_epoch_done(week)

            for consumer in tqdm(self.consumers, file = sys.stdout):
                simulated = consumer.create_basket(week, basket_id)
                for key,value in simulated.items():
                    dict_data[key].extend(value)
                basket_id += 1
        return dict_data

    def convert_to_dict(self):
        self.INFO_writing_to_file()

        data = self.simulate_consumer()
        df = pd.DataFrame.from_dict(data)
        df[:1000000].to_excel("large_data/simulated_data1.xlsx")
        df[1000000:2000000].to_excel("large_data/simulated_data2.xlsx")
        df[2000000:3000000].to_excel("large_data/simulated_data3.xlsx")
        df[3000000:].to_excel("large_data/simulated_data4.xlsx")

        self.INFO_completed_simulation()


    @staticmethod
    def create_correlation_matrix():
            PATH = get_filepath('resources', 'correlation_matrix_excel.xlsx')
            data = pd.read_excel(PATH, header=None)
            correlation_matrix = data.to_numpy()
            return correlation_matrix


class Consumer(object):

    def __init__(self, C, gamma, product_pref, J_c, p_sentitivity, dev, id):
        self.dev = dev
        self.p_sent = p_sentitivity
        self.J_c = J_c
        self.product_pref = product_pref
        self.gamma = gamma
        self.C = C
        self.has_id = False
        self.id = id
        self.omegas = self.create_omega_c()
        self.sigmas = self.create_sigma_c()
        self.category_prices = self.create_category_prices(0.5, 0.3)
        self.correlation_matrix = self.create_correlation_matrix()

    def create_omega_c(self):
        omegas = {}
        for i in range(self.C):
            omegas[i] = self.vine_method(self, self.J_c, 0.2, 1)
        return omegas

    def create_sigma_c(self):
        sigmas = {}
        I = np.eye(self.J_c)
        for i, omega in self.omegas.items():
            sigma_c = (self.product_pref * I) @ omega @ (self.product_pref * I)
            sigmas[i] = sigma_c
        return sigmas

    def create_category_prices(self, mean, std):
        prices = np.zeros(self.C)
        for i in range(prices.size):
            prices[i] = np.random.lognormal(mean, std)
        return prices

    def create_product_prices(self, category):
        product_prices = np.zeros(self.J_c)
        category_price = self.category_prices[category]
        for i in range(product_prices.size):
            product_prices[i] = np.random.uniform(category_price / 2, category_price * 2)
        return product_prices

    def create_base_utility(self, category):
        return self.draw_multivariate_normal(self, np.zeros(self.J_c), self.omegas[category])

    def choose_categories(self):
        categories = np.zeros(self.C)
        error = self.draw_multivariate_normal(self, np.zeros(self.C), self.correlation_matrix)
        for i in range(categories.size):
            utility = self.gamma + error[i]
            if utility > 0:
                categories[i] = 1
        return categories

    def buy_products(self, category):
        random = np.random.uniform(0,1)
        max_utility = (float('-inf'), -1)
        second_max_utility = (float('-inf'), -1)
        base_utility = self.create_base_utility(category)
        product_prices = self.create_product_prices(category)

        if random <= .50:
            for i in range(self.J_c):
                utility = base_utility[i] - self.p_sent * product_prices[i] + np.random.normal(0, self.dev)
                if utility > max_utility[0]:
                    second_max_utility = max_utility
                    max_utility = utility, i
                elif utility > second_max_utility[0]:
                    second_max_utility = utility, i
            return [max_utility[1] + category * self.J_c, second_max_utility[1] + category * self.J_c], [product_prices[max_utility[1]], product_prices[second_max_utility[1]]]

        else:
            for i in range(self.J_c):
                utility = base_utility[i] - self.p_sent * product_prices[i] + np.random.normal(0, self.dev)
                if utility > max_utility[0]:
                    max_utility = utility, i
            return [max_utility[1] + category * self.J_c], [product_prices[max_utility[1]]]

    def create_basket(self, week, basket_id):
        data = {}
        data.setdefault("i", [])
        data.setdefault("j", [])
        data.setdefault("price", [])
        data.setdefault("t", [])
        data.setdefault("basket_id", [])
        data.setdefault("category", [])
        categories = self.choose_categories()
        for i, choice in enumerate(categories):
            if choice == 1:
                products, price = self.buy_products(i)
                for x, product in enumerate(products):
                    data["i"].append(self.id)
                    data["j"].append(product)
                    data["price"].append(price[x])
                    data["t"].append(week)
                    data["basket_id"].append(basket_id)
                    data["category"].append(i)

        return data

    def get_id(self):
        return self.id


    @staticmethod
    # TODO: VALIDATE CORRECTNESS OF CORRELATION MATRIX
    def vine_method(self, dimensions, beta1, beta2):
        partial_correlations = np.zeros((dimensions, dimensions))
        omega_c = np.eye(dimensions)

        for k in range(dimensions):
            for i in range(k + 1, dimensions):
                partial_correlations[k, i] = np.random.beta(beta1, beta2)
                p = partial_correlations[k, i]
                for l in range(k, 0, -1):
                    p = p * np.math.sqrt(
                        (1 - partial_correlations[l, i] ** 2) * (1 - partial_correlations[l, k] ** 2)) + \
                        partial_correlations[l, i] * partial_correlations[l, k]
                omega_c[k, i] = p
                omega_c[i, k] = p
        return omega_c

    @staticmethod
    def create_correlation_matrix():
        PATH = get_filepath('resources', 'correlation_matrix_excel.xlsx')
        data = pd.read_excel(PATH, header=None)
        correlation_matrix = data.to_numpy()
        return correlation_matrix

    @staticmethod
    # TODO FIX ERROR OF POSITIVE DEFINITE MATRIX
    def draw_multivariate_normal(self, mean, correlation):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.random.multivariate_normal(mean, correlation)


if __name__ == "__main__":
    sim = basket_simulator(C=20, J_c=15, T=20, I = 1, num_consumers=20000)
    sim.convert_to_dict()
