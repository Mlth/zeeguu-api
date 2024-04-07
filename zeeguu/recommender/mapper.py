import os

from pandas import DataFrame
from zeeguu.core.model.article import Article
from zeeguu.core.model.user import User
import pickle

mappings_path = "./zeeguu/recommender/mappings/"
user_order_to_id_path = f"{mappings_path}user_order_mapping.pkl"
user_id_to_order_path = f"{mappings_path}user_id_mapping.pkl"
article_order_to_id_path = f"{mappings_path}article_order_mapping.pkl"
article_id_to_order_path = f"{mappings_path}article_id_mapping.pkl"

class Mapper:
    num_users = 0
    num_articles = 0

    def __init__(self):
        self.user_order_to_id = {}
        self.user_id_to_order = {}
        self.article_order_to_id = {}
        self.article_id_to_order = {}

        self.set_user_order_to_id()
        self.set_article_order_to_id()

    def set_article_order_to_id(self):
        if os.path.exists(article_order_to_id_path) and os.path.exists(article_id_to_order_path):
            print("Loading article mappings from files.")
            self.article_id_to_order = pickle.load(open(article_id_to_order_path, 'rb'))
            self.article_order_to_id = pickle.load(open(article_order_to_id_path, 'rb'))
            self.num_articles = len(self.article_order_to_id)
        else:
            print("No article mappings found. Building new mappings.")
            articles = Article.query.filter(Article.broken != 1).all()
            index = 0
            for article in articles:
                self.article_order_to_id[index] = article.id
                self.article_id_to_order[article.id] = index
                index += 1
            self.num_articles = index

            with open(article_order_to_id_path, 'wb') as f:
                pickle.dump(self.article_order_to_id, f)
            with open(article_id_to_order_path, 'wb') as f:
                pickle.dump(self.article_id_to_order, f)

    def set_user_order_to_id(self):
        if os.path.exists(user_order_to_id_path) and os.path.exists(user_id_to_order_path):
            print("Loading user mappings from files.")
            self.user_id_to_order = pickle.load(open(user_id_to_order_path, 'rb'))
            self.user_order_to_id = pickle.load(open(user_order_to_id_path, 'rb'))
            self.num_users = len(self.user_order_to_id)
        else:
            print("No user mappings found. Building new mappings.")
            users = User.query.filter(User.is_dev == False).all()
            index = 0
            for user in users:
                self.user_order_to_id[index] = user.id
                self.user_id_to_order[user.id] = index
                index += 1
            self.num_of_users = index

            with open(user_order_to_id_path, 'wb') as f:
                pickle.dump(self.user_order_to_id, f)
            with open(user_id_to_order_path, 'wb') as f:
                pickle.dump(self.user_id_to_order, f)

    def map_articles(self, articles: DataFrame):
        articles['id'] = articles['id'].map(self.article_id_to_order)
        return articles