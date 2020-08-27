import numpy as np
import os
import pickle
import random
from tqdm import tqdm
from sklearn.utils import shuffle

class Dataset(object):
    def __init__(self, BATCH_SIZE, dataset='douban'):
        self.batch_size = BATCH_SIZE
        self.peo2item_movie, self.peo2item_book, self.item2peo_movie, self.item2peo_book= self.load_data(dataset)
        self.movie_set = self.item2peo_movie.keys()
        self.book_set = self.item2peo_book.keys()
        self.num_user = len(self.peo2item_movie)
        self.num_movie = len(self.item2peo_movie)
        self.num_book = len(self.item2peo_book)
        self.movie_vali, self.movie_test, self.movie_nega, self.book_vali, self.book_test, self.book_nega = self.get_test_data(dataset)

        print('load data down')

        self.test_count = 0
        self.batch_count = 0
        self.count = 0
        self.epoch = 1

    def load_data(self, dataset):
        print('loading data...{}')
        peo2movie = pickle.load(open(os.path.join('processed_data', dataset,'peo2movie_id.pkl'), 'rb'))
        peo2book = pickle.load(open(os.path.join('processed_data', dataset,'peo2book_id.pkl'), 'rb'))
        movie2peo = pickle.load(open(os.path.join('processed_data', dataset,'movie2peo_id.pkl'), 'rb'))
        book2peo = pickle.load(open(os.path.join('processed_data', dataset,'book2peo_id.pkl'), 'rb'))

        return peo2movie, peo2book, movie2peo, book2peo

    # def get_train_indices(self, domain):
    #     row, col = [], []
    #
    #     if domain == 'movie':
    #         dict = self.peo2item_movie
    #         vali = self.movie_vali
    #     elif domain == 'book':
    #         dict = self.peo2item_book
    #         vali = self.book_vali
    #     else:
    #         return
    #
    #     for user in dict:
    #         dict[user].remove(vali[user])
    #         for each in dict[user]:
    #             row.append(user)
    #             col.append(each)
    #
    #     row = np.array(row)
    #     col = np.array(col)
    #
    #     return row, col

    def get_part_train_indices(self, domain, percent):
        row, col = [], []

        if domain == 'movie':
            dict = self.peo2item_movie
            vali = self.movie_vali
            test = self.movie_test
        elif domain == 'book':
            dict = self.peo2item_book
            vali = self.book_vali
            test = self.book_test
        else:
            return

        for user in dict:
            if len(dict[user])>2:
                dict[user].remove(vali[user])
                dict[user].remove(test[user])
            else:
                dict[user].remove(vali[user])

            dict[user] = shuffle(dict[user], random_state=72)
            num_item = int(round(percent * len(dict[user])))

            for i in range(num_item):
                row.append(user)
                col.append(dict[user][i])

        row = np.array(row)
        col = np.array(col)

        return row, col

    def get_test_data(self, dataset):
        if not os.path.exists(os.path.join(os.getcwd(), 'processed_data', dataset, 'movie_vali.pkl')) or \
                not os.path.exists(os.path.join(os.getcwd(), 'processed_data', dataset, 'movie_test.pkl')):
            movie_vali = {}
            movie_test = {}
            book_vali = {}
            book_test = {}
            movie_nega = {}
            book_nega = {}
            for i in tqdm(self.peo2item_movie):
                items = self.peo2item_movie[i]
                if len(items)>=2:
                    items_choices = shuffle(items, random_state=2020)[:2]
                    movie_vali[i] = items_choices[0]
                    movie_test[i] = items_choices[1]
                else:
                    movie_vali[i] = items[0]
                    movie_test[i] = items[0]

                all_nega_movies = list((set(list(range(self.num_movie))) - set(items)))
                movie_nega[i] = shuffle(all_nega_movies, random_state=2020)[:99]

            for i in tqdm(self.peo2item_book):
                items = self.peo2item_book[i]
                if len(items)>=2:
                    items_choices = shuffle(items, random_state=2020)[:2]
                    book_vali[i] = items_choices[0]
                    book_test[i] = items_choices[1]
                else:
                    book_vali[i] = items[0]
                    book_test[i] = items[0]

                all_nega_books = list((set(list(range(self.num_book))) - set(items)))
                book_nega[i] = shuffle(all_nega_books, random_state=2020)[:99]

            pickle.dump(movie_vali, open(os.path.join('processed_data', dataset, 'movie_vali.pkl'), 'wb'))
            pickle.dump(book_vali, open(os.path.join('processed_data', dataset, 'book_vali.pkl'), 'wb'))

            pickle.dump(movie_test, open(os.path.join('processed_data', dataset, 'movie_test.pkl'), 'wb'))
            pickle.dump(book_test, open(os.path.join('processed_data', dataset, 'book_test.pkl'), 'wb'))

            pickle.dump(movie_nega, open(os.path.join('processed_data', dataset, 'movie_nega.pkl'), 'wb'))
            pickle.dump(book_nega, open(os.path.join('processed_data', dataset, 'book_nega.pkl'), 'wb'))

        else:
            movie_vali = pickle.load(open(os.path.join('processed_data', dataset, 'movie_vali.pkl'), 'rb'))
            book_vali = pickle.load(open(os.path.join('processed_data', dataset, 'book_vali.pkl'), 'rb'))

            movie_test = pickle.load(open(os.path.join('processed_data', dataset, 'movie_test.pkl'), 'rb'))
            book_test = pickle.load(open(os.path.join('processed_data', dataset, 'book_test.pkl'), 'rb'))

            movie_nega = pickle.load(open(os.path.join('processed_data', dataset, 'movie_nega.pkl'), 'rb'))
            book_nega = pickle.load(open(os.path.join('processed_data', dataset, 'book_nega.pkl'), 'rb'))

        return movie_vali, movie_test, movie_nega, book_vali, book_test, book_nega



if __name__ == '__main__':
    dataset = Dataset(1000, 4, True)
    i = 1
    vali = dataset.movie_vali[i]
    nega = dataset.movie_nega[i]
    nega_vali = np.concatenate([nega, vali],1)






