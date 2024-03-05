from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.utils import ShowData

import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class Trainer:
    matrix_config = FeedbackMatrixConfig(
        show_data=[ShowData.ALL],
        adjustment_config=AdjustmentConfig(
            difficulty_weight=1,
            translation_adjustment_value=4
        ),
    )
    matrix = FeedbackMatrix(matrix_config)

    # Define embedding layers for users and items
    num_users = 1000  # Example: total number of users
    num_items = 500  # Example: total number of items
    embedding_dim = 50  # Example: size of embedding vectors

    def sparse_mean_square_error(self, sparse_ratings, user_embeddings, movie_embeddings):
        """
        Args:
            sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
            user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
            dimension, such that U_i is the embedding of user i.
            movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
            dimension, such that V_j is the embedding of movie j.
        Returns:
            A scalar Tensor representing the MSE between the true ratings and the
            model's predictions.
        """
        predictions = tf.reduce_sum(
            tf.gather(user_embeddings, sparse_ratings.indices[:, 0]) *
            tf.gather(movie_embeddings, sparse_ratings.indices[:, 1]),
            axis=1)
        loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
        return loss