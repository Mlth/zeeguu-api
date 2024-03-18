from enum import Enum, auto
import numpy as np
from pandas import DataFrame
from zeeguu.core.model.article import Article
from zeeguu.recommender.cf_model import CFModel
from zeeguu.recommender.tensor_utils import build_liked_sparse_tensor
from zeeguu.recommender.utils import ShowData, setup_df_rs
import pandas as pd
from IPython import display
from zeeguu.recommender.utils import get_resource_path
from zeeguu.recommender.tensor_utils_mock import build_mock_sparse_tensor, genereate_100_articles_with_titles, setup_sessions

import tensorflow as tf

from zeeguu.recommender.visualization.model_visualizer import ModelVisualizer
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class Measure(Enum):
    # If no ShowData is chosen, all data will be retrieved and shown.
    DOT = 'dot'
    COSINE = 'cosine'

class RecommenderSystem:
    cf_model = None
    visualizer = ModelVisualizer()

    def __init__(self, sessions, num_users, num_items, embedding_dim=20, stddev=1., test=False):
        self.num_users = num_users
        self.num_items = num_items
        self.sessions = sessions
        self.embedding_dim = embedding_dim
        self.stddev = stddev
        self.test=test
        if(test):
            print("warring running in test mode")
            self.articles = genereate_100_articles_with_titles()
        else:
            self.articles = setup_df_rs(self.num_items)

    def split_dataframe(self, df: DataFrame, holdout_fraction=0.1):
        """Splits a DataFrame into training and test sets.
        Args:
            df: a dataframe.
            holdout_fraction: fraction of dataframe rows to use in the test set.
        Returns:
            train: dataframe for training
            test: dataframe for testing
        """
        test = df.sample(frac=holdout_fraction, replace=False)
        train = df[~df.index.isin(test.index)]
        return train, test

    def sparse_mean_square_error(self, sparse_sessions, user_embeddings, article_embeddings):
        """
        Args:
            sparse_sessions: A SparseTensor rating matrix, of dense_shape [N, M]
            user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
            dimension, such that U_i is the embedding of user i.
            article_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
            dimension, such that V_j is the embedding of movie j.
        Returns:
            A scalar Tensor representing the MSE between the true ratings and the
            model's predictions.
        """
        predictions = tf.reduce_sum(
            tf.gather(user_embeddings, sparse_sessions.indices[:, 0]) *
            tf.gather(article_embeddings, sparse_sessions.indices[:, 1]),
            axis=1)
        loss = tf.losses.mean_squared_error(sparse_sessions.values, predictions)
        return loss
    
    def build_model(self):
        """
        Args:
            embedding_dim: the dimension of the embedding vectors.
            init_stddev: float, the standard deviation of the random initial embeddings.
        Returns:
            model: a CFModel.
        """
        # Split the sessions DataFrame into train and test.
       

        # SparseTensor representation of the train and test datasets.
        if(self.test):
            sessions = setup_sessions() # this is from the mocking file
            train_sessions, test_sessions = self.split_dataframe(sessions)
            A_train = build_mock_sparse_tensor(train_sessions, "train")
            A_test = build_mock_sparse_tensor(test_sessions, "test")
        else:
            train_sessions, test_sessions = self.split_dataframe(self.sessions)
            A_train = build_liked_sparse_tensor(train_sessions, self.num_users, self.num_items)
            A_test = build_liked_sparse_tensor(test_sessions, self.num_users, self.num_items)

        user_embeddings = tf.Variable(
            tf.fill(
                [self.num_users, self.embedding_dim], 0.2))#stddev=self.stddev))
        article_embeddings = tf.Variable(
            tf.fill(
                [self.num_items, self.embedding_dim], 0.2))#stddev=self.stddev))

        train_loss = self.sparse_mean_square_error(A_train, user_embeddings, article_embeddings)
        test_loss = self.sparse_mean_square_error(A_test, user_embeddings, article_embeddings)
        metrics = {
            'train_error': train_loss,
            'test_error': test_loss
        }
        embeddings = {
            "user_id": user_embeddings,
            "article_id": article_embeddings
        }
        self.cf_model = CFModel(embeddings, train_loss, [metrics])

    def compute_scores(self, query_embedding, item_embeddings, measure=Measure.DOT):
        """Computes the scores of the candidates given a query.
        Args:
            query_embedding: a vector of shape [k], representing the query embedding.
            item_embeddings: a matrix of shape [N, k], such that row i is the embedding
            of item i.
            measure: a string specifying the similarity measure to be used. Can be
            either DOT or COSINE.
        Returns:
            scores: a vector of shape [N], such that scores[i] is the score of item i.
        """
        u = query_embedding
        V = item_embeddings
        if measure == Measure.COSINE:
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            u = u / np.linalg.norm(u)
        scores = u.dot(V.T)
        return scores
    
    def user_recommendations(self, user_id, measure=Measure.DOT, exclude_rated=False, k=100):
        # TODO: Does user have (enough) interactions for us to be able to make accurate recommendations?
        should_recommend = True

        if should_recommend:
            scores = self.compute_scores(
                self.cf_model.embeddings["user_id"][user_id], self.cf_model.embeddings["article_id"], measure)
            score_key = str(measure) + ' score'
            df = pd.DataFrame({
                score_key: list(scores),
                'article_id': self.articles['id'],
                'titles': self.articles['title'],
            }).dropna(subset=["titles"])
            if exclude_rated:
                # remove articles that have already been read
                read_articles = self.sessions[self.sessions.user_id == user_id]["article_id"].values
                df = df[df.article_id.apply(lambda article_id: article_id not in read_articles)]
            display.display(df.sort_values([score_key], ascending=False).head(k))
        else:
            # Possibly do elastic stuff to just give some random recommendations
            return
        
    def visualize_article_embeddings(self):
        self.visualizer.visualize_tsne_article_embeddings(self.cf_model, self.articles)

    '''def movie_neighbors(model, title_substring, measure=Measure.DOT, k=6):
        # Search for movie ids that match the given substring.
        ids =  movies[movies['title'].str.contains(title_substring)].index.values
        titles = movies.iloc[ids]['title'].values
        if len(titles) == 0:
            raise ValueError("Found no movies with title %s" % title_substring)
        print("Nearest neighbors of : %s." % titles[0])
        if len(titles) > 1:
            print("[Found more than one matching movie. Other candidates: {}]".format(
                ", ".join(titles[1:])))
        movie_id = ids[0]
        scores = compute_scores(
            model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"],
            measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'titles': movies['title'],
            'genres': movies['all_genres']
        })
        display.display(df.sort_values([score_key], ascending=False).head(k))'''