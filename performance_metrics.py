from collections import defaultdict

import matplotlib
import scipy.spatial.distance
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly
import plotly.graph_objs as go

from tensorflow.python.keras.models import load_model

import sklearn.manifold
import sklearn.cluster
from data_pipeline import ExcelReader
from helper_functions import create_html, get_filepath


class evaluatePerformance:
    def __init__(self, model, metrics={}, product_file_dir=None, product_file_loc=None, product_used_loc=None):
        self.model_loc = model
        self.metrics = metrics
        try:
            self.product_info = self.read_product_data_simulated(product_file_dir, product_file_loc)
            self.products_used = set(np.loadtxt(product_used_loc, delimiter=",", dtype=np.int32))

        except FileNotFoundError and TypeError:
            print("{} ERROR: Unable to load the product_info or/and product_used files".format(
                datetime.now().strftime("%H:%M:%S")))

    @staticmethod
    def read_product_data_simulated(file_dir, file_location):
        data_reader = ExcelReader(4)
        simulated_data = data_reader.read_split_excel(file_location, file_dir)
        return simulated_data

    def read_product_data_instacart(self):
        raise NotImplementedError

    def get_input_weights(self, type_weights='center_embedding'):
        model = load_model("Models/" + self.model_loc)
        embedding_layer = model.get_layer(type_weights)
        weights = np.array(embedding_layer.get_weights()[0])
        weights /= np.linalg.norm(weights, axis=1)[:, np.newaxis]
        sorted_products = sorted(self.products_used)
        p2v_vectors = []
        for product in sorted_products:
            p2v_vectors.append(weights[product])
        return np.array(p2v_vectors), sorted_products

    def simulate_random_weights(self, mean, std, num_weights, size_vector):
        random_weights = np.random.normal(loc=mean, scale=std, size=(num_weights, size_vector))
        return random_weights

    @staticmethod
    def get_tsne_embedding(weights, official_run=False):
        if not official_run:
            tsne_output = sklearn.manifold.TSNE(random_state=1,
                                                n_components=2,
                                                n_iter=4000,
                                                perplexity=15,
                                                angle=.5,
                                                verbose=1).fit_transform(weights)
            return tsne_output

        else:
            best_embedding = None
            for i in range(50):
                if i % 10 == 0:
                    print('10 tsne mappings done')

                best_divergence = float('inf')
                tsne_model = sklearn.manifold.TSNE(n_components=2,
                                                   n_iter=4000,
                                                   perplexity=15,
                                                   angle=.5,
                                                   verbose=0)
                tsne_output = tsne_model.fit_transform(weights)
                divergence = tsne_model.kl_divergence_
                if best_divergence > divergence:
                    best_embedding = tsne_output
            return best_embedding

    def format_data(self, tsne_output):
        tsne_data = []
        for product in self.products_used:
            x = tsne_output[product][0]
            y = tsne_output[product][1]
            rows = self.product_info.loc[self.product_info['j'] == product].head(1)
            category = rows['category'].values[0]
            tsne_data.append(([x, y, category, product]))
        return pd.DataFrame(tsne_data, columns=['x', 'y', 'c', 'j'])

    def create_scatterplot(self, tsne_df, plot_file_name):
        # Function taken from: https://github.com/sbstn-gbl to enable comparison of the plots made
        plot_data = [go.Scatter(x=tsne_df['x'].values,
                                y=tsne_df['y'].values,
                                text=[
                                    'category = %d <br> product = %d' % (x, y)
                                    for (x, y) in zip(tsne_df['c'].values, tsne_df['j'].values)
                                ],
                                hoverinfo='text',
                                mode='markers',
                                marker=dict(
                                    size=14,
                                    color=tsne_df['c'].values,
                                    colorscale='Jet',
                                    showscale=False
                                )
                                )
                     ]
        legend = go.layout.Legend()
        plot_layout = go.Layout(
            width=800,
            height=600,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=4),
            hovermode='closest',
            legend=legend,
            showlegend=True
        )

        # plot
        fig = go.Figure(data=plot_data, layout=plot_layout)
        iplot(fig)

        plt.show()

    def category_level_similarity(self, num_product_in_category, vectors):
        category_to_assign = 0
        category_product = defaultdict(list)
        p2v_vectors = self.get_input_weights()
        acc = 0
        category = 0
        average_vectors = {}
        summed_vec = np.zeros(vectors.shape[1])
        for product in self.products_used:
            summed_vec += vectors[product]
            if acc == num_product_in_category - 1:
                acc = 0
                average_vectors[category] = summed_vec / num_product_in_category
                category += 1
                summed_vec = np.zeros(vectors.shape[1])
            else:
                acc += 1
        return average_vectors

    @staticmethod
    def similarity_matrix(average_vectors, filename):
        num_cat = len(average_vectors.keys())
        similarity = np.zeros((num_cat, num_cat))
        for i in range(num_cat):
            for j in range(num_cat):
                similarity[i][j] = average_vectors[i] @ average_vectors[j]
        create_html(similarity, filename)
        f = plt.figure(figsize=(19, 15))
        plt.matshow(similarity, fignum=f.number)
        plt.xticks(range(similarity.shape[1]), fontsize=14, rotation=90)
        plt.yticks(range(similarity.shape[1]), fontsize=14)
        cb = plt.colorbar()
        cb.set_clim(-1, 1)
        cb.ax.tick_params(labelsize=14)
        plt.show()

    def co_occurence_matrix(self, average_vectors_co, average_vectors_ce, filename):
        num_cat = len(average_vectors_ce.keys())
        similarity = np.zeros((num_cat, num_cat))
        for i in range(num_cat):
            for j in range(i, num_cat):
                similarity[i][j] = average_vectors_ce[j] @ average_vectors_co[i]
                similarity[j][i] = average_vectors_ce[j] @ average_vectors_co[i]

        create_html(similarity, filename)
        f = plt.figure(figsize=(19, 15))
        plt.matshow(similarity, fignum=f.number)
        plt.xticks(range(similarity.shape[1]), fontsize=14, rotation=90)
        plt.yticks(range(similarity.shape[1]), fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.show()


    def bench_marks(self, data):
        x_y_df = data.set_index(['j', 'c'])

        x_y_array = x_y_df.to_numpy()
        true_clusters = data['c'].to_numpy()

        clustering = sklearn.cluster.KMeans(n_clusters=len(np.unique(true_clusters)), n_init=30)
        kmean_pred = clustering.fit_predict(x_y_df.values)
        s_score = sklearn.metrics.silhouette_score(X=x_y_array, labels=true_clusters)
        adjusted_mis = sklearn.metrics.adjusted_mutual_info_score(
            labels_true=true_clusters,
            labels_pred=kmean_pred,
            average_method='arithmetic'
        )
        nn_score = self.get_hitrate(data, 14)

        return s_score, adjusted_mis, nn_score

    def get_hitrate(self, x, num_nn):
        xy = x[['x', 'y']].to_numpy()
        j = x['j'].to_numpy()
        c = x['c'].to_numpy()

        distances = scipy.spatial.distance.cdist(xy, xy)
        distance_df = pd.DataFrame({

            'j': np.repeat(j, len(j)),

            'c': np.repeat(c, len(c)),

            'j2': np.tile(j, len(j)),

            'c2': np.tile(c, len(c)),

            'd': distances.flatten()

        })
        distance_df = distance_df[distance_df['j'] != distance_df['j2']]

        distance_df = distance_df.sort_values('d')

        distance_df['rank_d'] = distance_df.groupby('j').cumcount()

        nn = distance_df[distance_df['rank_d'] < num_nn]
        score = float(sum(nn['c'] == nn['c2'])) / nn.shape[0]

        return score

    def create_histogram(self):
        correlation = self.product_info.corr()
        self.product_info.hist(column='j')
        plt.show()


def main():
    model_used = 'simulated_test.h5'
    data_dir = 'large_data'
    data_file = 'simulated_data'
    product_used = r'large_data\center_products_simulated'
    performance_logger = evaluatePerformance(model=model_used, product_file_dir=data_dir,
                                             product_file_loc=data_file,
                                             product_used_loc=product_used)
    weights, sorted_products = performance_logger.get_input_weights()
    #co_occurence_weights, sorted_products = performance_logger.get_input_weights(type_weights='context_embedding')
    random_weights = performance_logger.simulate_random_weights(0, .2, weights.shape[0], weights.shape[1])

    print("Starting on center weights")
    tsne_ce = performance_logger.get_tsne_embedding(weights, official_run=True)
    dank_df = performance_logger.format_data(tsne_ce)
    s_score, info_score, nn_score = performance_logger.bench_marks(dank_df)
    av_vec_ce = performance_logger.category_level_similarity(15, weights)
    performance_logger.similarity_matrix(av_vec_ce, 'similarity_correlation.html')

    print("Starting on random weights")
    tsne_r = performance_logger.get_tsne_embedding(random_weights)
    dank_df_r = performance_logger.format_data(tsne_r)
    s_score_r, info_score_r, nn_score_r = performance_logger.bench_marks(dank_df_r)
    av_vec = performance_logger.category_level_similarity(15, random_weights)
    performance_logger.similarity_matrix(av_vec, 'random_weights.html')
    """
    print("Starting on context embeddings")
    tsne_co = performance_logger.get_tsne_embedding(co_occurence_weights, official_run=True)
    dank_df_c = performance_logger.format_data(tsne_co)
    s_score_c, info_score_c, nn_score_c = performance_logger.bench_marks(dank_df_c)
    av_vec_co = performance_logger.category_level_similarity(15, co_occurence_weights)
    performance_logger.co_occurence_matrix(av_vec_co, av_vec_ce, 'co-occurence_correlation.html')"""

    print("Metrics for trained weights, s_score: {}, adj_info: {}, nn_score: {}".format(s_score, info_score,
                                                                                        nn_score))
    print("Metrics for random weights, s_score: {}, adj_info: {}, nn_score: {}".format(s_score_r, info_score_r,
                                                                                       nn_score_r))
    #print("Metrics for contex weights, s_score: {}, adj_info: {}, nn_score: {}".format(s_score_c, info_score_c, nn_score_c))

    performance_logger.create_scatterplot(dank_df, 'test')
    #performance_logger.create_scatterplot(dank_df_c, 'test')

    def create_correlation_matrix():
            PATH = get_filepath('resources', 'correlation_matrix_excel.xlsx')
            data = pd.read_excel(PATH, header=None)
            correlation_matrix = data.to_numpy()

            f = plt.figure(figsize=(19, 15))
            plt.matshow(correlation_matrix, fignum=f.number)
            plt.xticks(range(correlation_matrix.shape[1]), fontsize=14, rotation=90)
            plt.yticks(range(correlation_matrix.shape[1]), fontsize=14)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.show()

    create_correlation_matrix()


if __name__ == '__main__':
    main()
