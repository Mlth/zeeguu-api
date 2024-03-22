import altair as alt
import sklearn
import sklearn.manifold
from zeeguu.recommender.utils import get_resource_path

class ModelVisualizer:
    def __visualize_article_embeddings(self, data, x, y):
        nearest = alt.selection_point(
            encodings=['x', 'y'], on='mouseover', nearest=True, empty=True)
        base = alt.Chart().mark_circle().encode(
            x=x,
            y=y,
        ).properties(
            width=600,
            height=600,
        ).add_params(nearest)
        text = alt.Chart().mark_text(align='left', dx=5, dy=-5).encode(
            x=x,
            y=y,
            text=alt.condition(nearest, 'title', alt.value('')))
        return alt.hconcat(alt.layer(base, text), data=data)
    
    def __tsne_article_embeddings(self, model, articles):
        """Visualizes the article embeddings, projected using t-SNE with Cosine measure.
        Args:
            model: A MFModel object.
        """
        tsne = sklearn.manifold.TSNE(
            n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
            init='pca', verbose=True, n_iter=400)

        print('Running t-SNE...')
        V_proj = tsne.fit_transform(model.embeddings["article_id"])
        articles.loc[:,'x'] = V_proj[:, 0]
        articles.loc[:,'y'] = V_proj[:, 1]
        return self.__visualize_article_embeddings(articles, 'x', 'y')
    
    def visualize_tsne_article_embeddings(self, model, articles):
        return self.__tsne_article_embeddings(model, articles).save(get_resource_path() + "article_embeddings.json")
