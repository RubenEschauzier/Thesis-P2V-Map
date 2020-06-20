import random
import numpy as np
from data_pipeline import ExcelReader, InstaCartPipeline


def create_input_file(baskets, train_loc, val_loc):
    max_len = 0
    for basket_id, basket in baskets.items():
        r_num = random.random()
        if r_num >= .1:
            if len(basket) >= 2:
                middle_index = len(basket) // 2
                split_arrays = np.split(np.array(basket), [middle_index])
                with open(train_loc, 'a') as f:
                    max_len = write_to_file(f, split_arrays, max_len)
        else:
            if len(basket) >= 2:
                middle_index = len(basket) // 2
                split_arrays = np.split(np.array(basket), [middle_index])
                with open(val_loc, 'a') as f:
                    max_len = write_to_file(f, split_arrays, max_len)

    return max_len


def write_to_file(f, split_arrays, max_len):
    final_product = split_arrays[0].shape[0]
    final_product1 = split_arrays[1].shape[0]
    if final_product1 > max_len:
        max_len = final_product1
    for i, element in enumerate(split_arrays[0]):
        if i == final_product - 1:
            f.writelines('%s\n' % element)
        else:
            f.writelines('%s ' % element)

    for j, element1 in enumerate(split_arrays[1]):
        if j == final_product1 - 1:
            f.writelines('%s\n\n' % element1)
        else:
            f.writelines('%s ' % element1)
    return max_len


def create_vocabulary(basket_in_data):
    products_in_basket = set()
    for basket_id, basket in basket_in_data.items():
        for element in basket:
            products_in_basket.add(element)
    with open('vocab.txt', 'a') as f:
        for product in products_in_basket:
            f.writelines('%s\n' % product)


if __name__ == '__main__':
    simulated_data = False
    pipeline = InstaCartPipeline(100, "order_products__train", "large_data", "order_id", "product_id")
    pipeline.filter_rare_observations()
    baskets = pipeline.get_baskets()
    if simulated_data:
        create_input_file(baskets, 'train_input_simulated.txt', 'val_input_simulated.txt')
    else:
        create_input_file(baskets, 'train_input_emperical.txt', 'val_input_emperical.txt')
