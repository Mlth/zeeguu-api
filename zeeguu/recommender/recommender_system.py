from enum import Enum
import numpy as np
from zeeguu.recommender.cf_model import CFModel
from zeeguu.recommender.tensor_utils import build_liked_sparse_tensor
from zeeguu.recommender.utils import filter_article_embeddings, get_recommendable_articles, setup_df_rs
import pandas as pd
from typing import Callable
from IPython import display
from zeeguu.recommender.mock.tensor_utils_mock import build_mock_sparse_tensor
from zeeguu.recommender.mock.generators_mock import generate_articles_with_titles
from zeeguu.recommender.visualization.model_visualizer import ModelVisualizer
from tensorflow.python.keras import layers
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
@tf.function(experimental_follow_type_hints=True)

class Measure(Enum):
    # If no ShowData is chosen, all data will be retrieved and shown.
    DOT = 'dot'
    COSINE = 'cosine'

class RecommenderSystem:
    cf_model = None
    visualizer = ModelVisualizer()
    user_embeddings_path = "./zeeguu/recommender/embeddings/user_embedding.npy"
    article_embeddings_path = "./zeeguu/recommender/embeddings/article_embedding.npy"

    def __init__(
        self,
        sessions : pd.DataFrame,
        num_users: int,
        num_items: int,
        embedding_dim : int =20,
        test=False,
        generator_function: Callable=None #function type
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.sessions = sessions
        self.embedding_dim = embedding_dim
        self.test=test
        self.generator_function = generator_function
        if(test):
            print("Warning! Running in test mode")
            self.articles = generate_articles_with_titles(num_items)
        else:
            self.articles = get_recommendable_articles()

    def split_dataframe(self, df: pd.DataFrame, holdout_fraction : float =0.1):
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


    def sparse_mean_square_error(self, sparse_sessions , user_embeddings, article_embeddings):
        """
        Args:
            sparse_sessions: A SparseTensor session matrix, of dense_shape [N, M]
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
    
    def save_embeddings(self, path):

        user_em = self.cf_model.embeddings["user_id"]
        article_em = self.cf_model.embeddings["article_id"]

        with open(path + "user_embedding.npy", 'wb' ) as f:
            np.save(f, user_em)

        with open(path + "article_embedding.npy", 'wb' ) as f:
            np.save(f, article_em)
    
    def build_model(self, stddev=1.0):
        """
        Args:
            embedding_dim: the dimension of the embedding vectors.
            init_stddev: float, the standard deviation of the random initial embeddings.
        Returns:
            model: a CFModel.
        """
        # SparseTensor representation of the train and test datasets.
        if(self.test):
            self.sessions = self.generator_function(self.num_users, self.num_items) 
            train_sessions, test_sessions = self.split_dataframe(self.sessions)
            A_train = build_mock_sparse_tensor(train_sessions, "train", self.num_users, self.num_items)
            A_test = build_mock_sparse_tensor(test_sessions, "test", self.num_users, self.num_items)
        else:
            train_sessions, test_sessions = self.split_dataframe(self.sessions)
            A_train = build_liked_sparse_tensor(train_sessions, self.num_users, self.num_items)
            A_test = build_liked_sparse_tensor(test_sessions, self.num_users, self.num_items)

        user_embeddings = tf.Variable(
            tf.random_normal(
                [self.num_users, self.embedding_dim], stddev=stddev))
        article_embeddings = tf.Variable(
            tf.random_normal(
                [self.num_items, self.embedding_dim], stddev=stddev))
        
        
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

    def build_regularized_model(self, regularization_coeff=.1, gravity_coeff=1., init_stddev=0.1):
        """
        Args:
            ratings: the DataFrame of movie ratings.
            embedding_dim: The dimension of the embedding space.
            regularization_coeff: The regularization coefficient lambda.
            gravity_coeff: The gravity regularization coefficient lambda_g.
        Returns:
            A CFModel object that uses a regularized loss.
        """

        if(self.test):
            self.sessions = self.generator_function(self.num_users, self.num_items) 
            train_sessions, test_sessions = self.split_dataframe(self.sessions)
            A_train = build_mock_sparse_tensor(train_sessions, "train", self.num_users, self.num_items)
            A_test = build_mock_sparse_tensor(test_sessions, "test", self.num_users, self.num_items)
        else:
            train_sessions, test_sessions = self.split_dataframe(self.sessions)
            A_train = build_liked_sparse_tensor(train_sessions, self.num_users, self.num_items)
            A_test = build_liked_sparse_tensor(test_sessions, self.num_users, self.num_items)

        user_embeddings = tf.Variable(tf.random_normal(
            [A_train.dense_shape[0], self.embedding_dim], stddev=init_stddev))
        article_embeddings = tf.Variable(tf.random_normal(
            [A_train.dense_shape[1], self.embedding_dim], stddev=init_stddev))

        error_train = self.sparse_mean_square_error(A_train, user_embeddings, article_embeddings)
        error_test = self.sparse_mean_square_error(A_test, user_embeddings, article_embeddings)
        gravity_loss = gravity_coeff * gravity(user_embeddings, article_embeddings)
        regularization_loss = regularization_coeff * (
            # The Colab notebook just summed the values of each embedding vector. Normally, the norm of a vector is calculated using the formula for Euclidian norm.
            # tf.reduce_sum(user_embeddings * user_embeddings) / user_embeddings.shape[0].value + tf.reduce_sum(article_embeddings * article_embeddings) / article_embeddings.shape[0].value)
            tf.norm(user_embeddings*user_embeddings)/user_embeddings.shape[0].value + tf.norm(article_embeddings*article_embeddings)/article_embeddings.shape[0].value)
        total_loss = error_train + regularization_loss + gravity_loss

        losses = {
            'train_error': error_train,
            'test_error': error_test
        }
        loss_components = {
            'observed_loss': error_train,
            'regularization_loss': regularization_loss,
            'gravity_loss': gravity_loss,
        }
        embeddings = {
            "user_id": user_embeddings,
            "article_id": article_embeddings
        }

        self.cf_model = CFModel(embeddings, total_loss, [losses, loss_components])

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
    
    def user_recommendations(self, user_id : int, measure=Measure.DOT, exclude_read: bool =False): #, k=10):
        user_likes = self.sessions[self.sessions["user_id"] == user_id]
        print(f"User likes: {user_likes['article_id']}")

        user_embeddings = np.load(self.user_embeddings_path)
        article_embeddings = np.load(self.article_embeddings_path)

        """ user_embeddings = self.cf_model.embeddings["user_id"]
        article_embeddings = self.cf_model.embeddings["article_id"] """

        # TODO: Does user have (enough) interactions for us to be able to make accurate recommendations?fe
        should_recommend = True
        if should_recommend:
            valid_article_embeddings = filter_article_embeddings(article_embeddings, self.articles['id'])
            scores = self.compute_scores(
                user_embeddings[user_id], valid_article_embeddings, measure)
            score_key = str(measure) + ' score'
            df = pd.DataFrame({
                score_key: list(scores),
                'article_id': self.articles['id'],
                #'titles': self.articles['title'],
            })#.dropna(subset=["titles"]) # dopna no longer needed because we filter in the articles that we save in the RecommenderSystem itself.
            if exclude_read:
                # remove articles that have already been read
                read_articles = self.sessions[self.sessions.user_id == user_id]["article_id"].values
                df = df[df.article_id.apply(lambda article_id: article_id not in read_articles)]
            display.display(df.sort_values([score_key], ascending=False).head(self.num_items))
        else:
            # Possibly do elastic stuff to just give some random recommendations
            return
        
    def article_neighbors(self, article_id, measure=Measure.DOT, k=10):
        scores = self.compute_scores(
            self.cf_model.embeddings["article_id"][article_id], self.cf_model.embeddings["article_id"],
            measure)
        score_key = str(measure) + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'article_id': self.articles['id'],
            'titles': self.articles['title'],
        })
        display.display(df.sort_values([score_key], ascending=False).head(k))

    def visualize_article_embeddings(self, marked_articles=[]):
        #TODO Fix for small test cases. Right now, the function crashes with low user/article count.
        self.visualizer.visualize_tsne_article_embeddings(self.cf_model, self.articles, marked_articles)

def gravity(U, V):
    """Creates a gravity loss given two embedding matrices."""
    return 1. / (U.shape[0].value*V.shape[0].value) * tf.reduce_sum(
        tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))