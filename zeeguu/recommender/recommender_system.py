from pandas import DataFrame
from zeeguu.recommender.cf_model import CFModel
from zeeguu.recommender.tensor_utils import build_liked_sparse_tensor
from zeeguu.recommender.utils import ShowData

import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class RecommenderSystem:
    cf_model = None
    user_embeddings = None
    article_embeddings = None

    def __init__(self, num_users, num_items, embedding_dim=50, stddev=1.):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.stddev = stddev

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
        predictions = tf.gather_nd(
            tf.matmul(user_embeddings, article_embeddings, transpose_b=True),
            sparse_sessions.indices)
        loss = tf.losses.mean_squared_error(sparse_sessions.values, predictions)
        return loss
    
    def build_model(self, liked_sessions_df):
        """
        Args:
            liked_sessions_df: a DataFrame of the liked sessions
            embedding_dim: the dimension of the embedding vectors.
            init_stddev: float, the standard deviation of the random initial embeddings.
        Returns:
            model: a CFModel.
        """
        # Split the sessions DataFrame into train and test.
        train_sessions, test_sessions = self.split_dataframe(liked_sessions_df)

        # SparseTensor representation of the train and test datasets.
        A_train = build_liked_sparse_tensor(train_sessions, self.num_users, self.num_items)
        A_test = build_liked_sparse_tensor(test_sessions, self.num_users, self.num_items)

        self.user_embeddings = tf.Variable(tf.random_normal([A_train.dense_shape[0], self.embedding_dim], stddev=self.stddev))
        self.article_embeddings = tf.Variable(tf.random_normal([A_train.dense_shape[1], self.embedding_dim], stddev=self.stddev))

        train_loss = self.sparse_mean_square_error(A_train, self.user_embeddings, self.article_embeddings)
        test_loss = self.sparse_mean_square_error(A_test, self.user_embeddings, self.article_embeddings)
        metrics = {
            'train_error': train_loss,
            'test_error': test_loss
        }
        embeddings = {
            "user_id": self.user_embeddings,
            "article_id": self.article_embeddings
        }
        self.cf_model = CFModel(embeddings, train_loss, [metrics])