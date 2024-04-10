from enum import Enum
import os
import numpy as np
from zeeguu.recommender.cf_model import CFModel
from zeeguu.recommender.mapper import Mapper
from zeeguu.recommender.utils.tensor_utils import build_liked_sparse_tensor
from zeeguu.recommender.utils.train_utils import Measure, train
from zeeguu.recommender.utils.recommender_utils import filter_article_embeddings, get_recommendable_articles, setup_df_rs
from zeeguu.recommender.utils.train_utils import user_embeddings_path, article_embeddings_path
import pandas as pd
from typing import Callable
from IPython import display
from zeeguu.recommender.mock.tensor_utils_mock import build_mock_sparse_tensor
from zeeguu.recommender.mock.generators_mock import generate_articles_with_titles
from zeeguu.recommender.visualization.model_visualizer import ModelVisualizer
from zeeguu.recommender.utils.elastic_utils import find_articles_like

class RecommenderSystem:
    visualizer = ModelVisualizer()

    def __init__(
        self,
        sessions : pd.DataFrame,
        mapper: Mapper,
        num_users: int,
        num_items: int,
        embedding_dim : int =20,
        generator_function: Callable=None, #function type
        stddev=0.1,
    ):
        self.mapper = mapper
        self.test=generator_function is not None
        if(self.test):
            print("Warning! Running in test mode")
            self.sessions = generator_function(num_users, num_items)
            self.articles = generate_articles_with_titles(num_items)
        else:
            self.sessions = sessions
            articles = get_recommendable_articles()
            self.articles = mapper.map_articles(articles)
        self.cf_model = CFModel(self.sessions, num_users, num_items, embedding_dim, self.test, stddev)

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
    
    def user_recommendations(self, user_id: int, language_id: int, measure=Measure.DOT, exclude_read: bool =False, k=None):
        if self.test:
            user_order = user_id
        else:
            user_order = self.mapper.user_id_to_order.get(user_id)
        user_likes = self.sessions[self.sessions["user_id"] == user_order]['article_id'].values
        print(f"User likes: {user_likes}")

        user_embeddings = self.cf_model.embeddings["user_id"]
        article_embeddings = self.cf_model.embeddings["article_id"]

        # TODO: Does user have (enough) interactions for us to be able to make accurate recommendations?
        should_recommend = True
        if should_recommend:
            valid_articles = self.articles[self.articles['language_id'] == language_id]
            valid_article_embeddings = filter_article_embeddings(article_embeddings, valid_articles['id'])
            scores = self.compute_scores(
                user_embeddings[user_order], valid_article_embeddings, measure)
            score_key = str(measure) + ' score'
            df = pd.DataFrame({
                score_key: list(scores),
                'article_id': valid_articles['id'],
                'language_id': valid_articles['language_id'],
                #'titles': valid_articles['title'],
            })#.dropna(subset=["titles"]) # dopna no longer needed because we filter in the articles that we save in the RecommenderSystem itself.
            if exclude_read:
                # remove articles that have already been read
                read_articles = self.sessions[self.sessions.user_id == user_order]["article_id"].values
                df = df[df.article_id.apply(lambda article_id: article_id not in read_articles)]
            df['article_id'] = df['article_id'].map(self.mapper.article_order_to_id)
            display.display(df.sort_values([score_key], ascending=False).head(len(df) if k is None else k))

            top_recommendations_with_total_likes = [f"{l}: {len(self.sessions[self.sessions['article_id'] == l]['article_id'].values)}" for l in df.sort_values([score_key], ascending=False).head(10)['article_id'].values]
            print(f"Total likes for top recommendations: {top_recommendations_with_total_likes}")
            
            top_ten = df.sort_values([score_key], ascending=False).head(10)['article_id'].values
            articles_to_recommend = find_articles_like(top_ten,5,30, language_id)
            print("this is what elastic thinks \n")
            for article in articles_to_recommend:
                print(article.title, article.language, article.published_time)
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