import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def build_mock_sparse_tensor(sessions,title):
    # Sort the indices
    sessions = sessions.sort_values(['user_id', 'article_id'])
    indices = sessions[['user_id', 'article_id']].values
    values = sessions['expected_read'].values

    tensor = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[100, 100]
    )

    dense_tensor = tf.sparse.to_dense(tensor)
    name = title + ".png"
    # Print the tensor within TensorFlow session
    with tf.compat.v1.Session() as sess:
        tensor_value = sess.run(dense_tensor)
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(tensor_value, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.title('Mock Tensor (100x100)')
        plt.xlabel('Article ID')
        plt.ylabel('User ID')
        plt.savefig(name)  # Save the plot to the specified filename
    
    print(f"Plot saved to {name}")
    
    return tensor

def setup_sessions() -> pd.DataFrame:
    users = pd.DataFrame()
    users['id'] = range(0, 100)
    users['name'] = [f'user_{i}' for i in range(0, 100)]
    '''This mocks sessions in a DataFrame where each user likes articles within 5 units of their user ID'''
    sessions_data = []
    for user_id in users['id']:
        # Determine the range of article IDs based on user ID
        min_article_id = user_id - 5
        if(min_article_id < 0):
            min_article_id = 0
        max_article_id = user_id + 5
        if(max_article_id > 99):
            max_article_id = 100
        
        liked_articles = [article_id for article_id in range(min_article_id, max_article_id)]        
        # Append user ID, article IDs within the range, and the expected read value to sessions_data
        for article_id in liked_articles:
            if(article_id != 50): #This is to show that everyone gets 50 recommended because none likes it 🤔🤔🤔
                sessions_data.append({'user_id': user_id, 'article_id': article_id, 'expected_read': 1.0})
    # Create a DataFrame from sessions_data
    sessions = pd.DataFrame(sessions_data)
    return sessions

def genereate_100_articles_with_titles() -> pd.DataFrame:
    '''This mocks 100 articles in a DataFrame'''
    articles = pd.DataFrame()
    articles['id'] = range(0, 100)
    articles['title'] = [f'article_{i}' for i in range(0, 100)]
    return articles