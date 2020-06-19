import json

import numpy as np
import pandas as pd
import sklearn.manifold
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly
import plotly.graph_objs as go

from data_pipeline import ExcelReader
from tensorflow.python.keras.models import load_model


def get_input_weights(model_name, products_used):
    model = load_model("Models/" + model_name)
    embedding_layer = model.get_layer('center_embedding')
    weights = np.array(embedding_layer.get_weights()[0])
    weights /= np.linalg.norm(weights, axis=1)[:, np.newaxis]
    sorted_products = sorted(products_used)
    p2v_vectors = []
    for product in sorted_products:
        p2v_vectors.append(weights[product])
    return np.array(p2v_vectors), sorted_products


def get_bert_vectors(file_dir, filename, num_products, size_vector, pca=True, random_state=1, n_components=30):
    bert_sum_vectors = json.load(open("{}/{}".format(file_dir, filename)))
    all_products = bert_sum_vectors.keys()
    int_products = []
    # Convert any numbers saved as string to integers and filter out any non integer strings
    for product_string in all_products:
        try:
            product = int(product_string)
            int_products.append(product)
        except ValueError:
            print('Non int token')
    # Sort products so our vector array is also sorted
    sorted_products = sorted(int_products)
    bert_vectors = []
    for token in sorted_products:
        try:
            product = int(token)
        except ValueError:
            print('Non int token')
        if product:
            vector_and_count = bert_sum_vectors[str(product)]
            bert_vector = np.array(vector_and_count[0]) / vector_and_count[1]
            bert_vectors.append(bert_vector)
    if pca:
        bert_vectors = sklearn.decomposition.PCA(random_state=1, n_components=n_components).fit_transform(bert_vectors)

    bert_vectors /= np.linalg.norm(bert_vectors, axis=1)[:, np.newaxis]

    return bert_vectors, sorted_products


def get_tsne_embedding(weights):
    tsne_output = sklearn.manifold.TSNE(random_state=1,
                                        n_components=2,
                                        n_iter=4000,
                                        perplexity=15,
                                        angle=.5,
                                        verbose=1).fit_transform(weights)
    return tsne_output


def create_plotting_df_instacart(tsne_output):
    # tsne_output = np.loadtxt(r'tsneoutput', delimiter=",", dtype=np.float32)
    product_data = pd.read_excel(r"large_data/products.xlsx")
    product_data.drop(product_data.columns[4:], axis=1, inplace=True)

    products_used = set(np.loadtxt(r'large_data\center_products_extra_filtered', delimiter=",", dtype=np.int32))
    # needs to be tested still!
    test_list = []
    for product in products_used:
        x = tsne_output[product][0]
        y = tsne_output[product][1]
        rows = product_data.loc[product_data['product_id'] == product]
        category = rows['department_id']
        try:
            temp_list = [x, y, int(category.values[0]), product]
            test_list.append(temp_list)
        except IndexError:
            print('Appears a weird artifact in the data exists')
        except ValueError:
            print(r'Thats no number!')

    tsne_df = pd.DataFrame(test_list, columns=['x', 'y', 'c', 'j'])
    return tsne_df


def create_plotting_df_simulated(tsne_output):
    data_reader = ExcelReader(4)
    simulated_data = data_reader.read_split_excel("simulated_data", "large_data")
    products_used = set(np.loadtxt(r'large_data\center_products_simulated', delimiter=",", dtype=np.int32))
    tsne_data = []
    for product in products_used:
        x = tsne_output[product][0]
        y = tsne_output[product][1]
        rows = simulated_data.loc[simulated_data['j'] == product].head(1)
        category = rows['category'].values[0]
        tsne_data.append(([x, y, category, product]))
    return pd.DataFrame(tsne_data, columns=['x', 'y', 'c', 'j'])


def create_plotting_df_bert(tsne_output, sorted_products):
    product_data = pd.read_excel(r"large_data/products.xlsx")
    product_data.drop(product_data.columns[4:], axis=1, inplace=True)
    data_frame_con_list = []
    for i, product in enumerate(sorted_products):
        x = tsne_output[i][0]
        y = tsne_output[i][1]
        rows = product_data.loc[product_data['product_id'] == product]
        category = rows['department_id']
        try:
            temp_list = [x, y, int(category.values[0]), product]
            data_frame_con_list.append(temp_list)
        except IndexError:
            print('Appears a weird artifact in the data exists')
        except ValueError:
            print(r'Thats no number!')

    tsne_df = pd.DataFrame(data_frame_con_list, columns=['x', 'y', 'c', 'j'])
    print(tsne_df)
    return tsne_df


def create_scatterplot(tsne_df, plot_file_name):
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
    plot_layout = go.Layout(
        width=800,
        height=600,
        margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=4),
        hovermode='closest'
    )

    # plot
    fig = go.Figure(data=plot_data, layout=plot_layout)
    plotly.offline.plot(fig, filename='testt')


if __name__ == '__main__':
    """bert_vectors_pca, sorted_products = get_bert_vectors('large_data', 'bert_product_vectors_vector_output_fixed.json',
                                                         50000, 768)
    tsne_output_bert = get_tsne_embedding(bert_vectors_pca)
    plotting_df = create_plotting_df_bert(tsne_output_bert, sorted_products)
    create_scatterplot(plotting_df)"""

    product_data = pd.read_excel(r"large_data/products.xlsx")
    product_data.drop(product_data.columns[4:], axis=1, inplace=True)

    products_used = set(np.loadtxt(r'large_data\center_products_extra_filtered', delimiter=",", dtype=np.int32))
    model_name = "instacart_model_final_filtered.h5"
    weights, sorted_products = get_input_weights(model_name, products_used)
    tsne_data = get_tsne_embedding(weights)
    df_to_plot = create_plotting_df_bert(tsne_data, sorted_products)
    create_scatterplot(df_to_plot, 'instacart_plot_extra')
