from zeeguu.core.model import Topic
from zeeguu.core.model.article import Article, article_topic_map
from zeeguu.core.model.difficulty_lingo_rank import DifficultyLingoRank
from elasticsearch import Elasticsearch, helpers
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX


def find_topics(article_id, session):
    article_topic = (
        session.query(Topic)
        .join(article_topic_map)
        .filter(article_topic_map.c.article_id == article_id)
    )
    topics = ""
    for t in article_topic:
        topics = topics + str(t.title) + " "

    return topics.rstrip()


def document_from_article(article, session):
    topics = find_topics(article.id, session)

    doc = {
        "title": article.title,
        "author": article.authors,
        "content": article.content,
        "summary": article.summary,
        "word_count": article.word_count,
        "published_time": article.published_time,
        "topics": topics,
        "language": article.language.name,
        "fk_difficulty": article.fk_difficulty,
        "lr_difficulty": DifficultyLingoRank.value_for_article(article),
        "url": article.url.as_string() if article.url else '', #fix for AttributeError: 'NoneType' object has no attribute 'as_string'
        "video": article.video,
    }
    return doc


def create_or_update(article, session):
    es = Elasticsearch(ES_CONN_STRING)

    doc = document_from_article(article, session)

    if es.exists(index=ES_ZINDEX, id=article.id):
       es.delete(index=ES_ZINDEX, id=article.id)
       
    res = es.index(index=ES_ZINDEX, id=article.id, body=doc)

    return res

def index_all_articles_helper(session):
    articles_to_index = Article.all_younger_than(0)
    for article in articles_to_index:
         doc = document_from_article(article, session)
         yield {
             "_index": ES_ZINDEX,
             "_id": article.id,
             "_source": doc,
         }
         
def index_all_articles(session):
    es = Elasticsearch(ES_CONN_STRING)
    helpers.bulk(es, index_all_articles_helper(session))

def index_in_elasticsearch(new_article, session):
    """
    # Saves the news article at ElasticSearch.
    # We recommend that everything is stored both in SQL and Elasticsearch
    # as ElasticSearch isn't persistent data
    """
    try:
        es = Elasticsearch(ES_CONN_STRING)
        doc = document_from_article(new_article, session)
        res = es.index(index=ES_ZINDEX, id=new_article.id, document=doc)

    except Exception as e:
        import traceback

        traceback.print_exc()


def remove_from_index(article):
    es = Elasticsearch(ES_CONN_STRING)
    if es.exists(index=ES_ZINDEX, id=article.id):
        print("Found in ES Index")
        es.delete(index=ES_ZINDEX, id=article.id)
        print("After deletion from the index.")
