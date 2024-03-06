import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

def build_liked_sparse_tensor(liked_sessions_df, num_of_users, num_of_articles):
    """
    Args:
        liked_sessions_df: a pd.DataFrame with `user_id`, `movie_id` and `expected_read` columns.
    Returns:
        A tf.SparseTensor representing the liked_sessions matrix.
    """
    indices = liked_sessions_df[['user_id', 'article_id']].values
    values = liked_sessions_df['expected_read'].values

    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[num_of_users, num_of_articles]
    )