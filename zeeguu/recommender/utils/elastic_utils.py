from datetime import timedelta, datetime
from zeeguu.core.model import Language
from zeeguu.core.model import Article, UserReadingSession, User
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from elasticsearch import Elasticsearch
from zeeguu.core.elastic.elastic_query_builder import build_elastic_recommender_query,build_elastic_search_query
from zeeguu.core.content_recommender.elastic_recommender import _to_articles_from_ES_hits, article_recommendations_for_user, candidate


def build_candidate_pool_for_lang(language: str,search: str = None, limit: int = None) -> 'list[candidate]':
    '''Input must be in lowercase short form
    Examples: en, da, ru.. without limit you get about 30.000 articles returned here, its optional
    call with true if you want to do a search, otherwise it will recommend'''
    lang = Language.find(language)
    if(limit!=None):
        es = Elasticsearch(ES_CONN_STRING)
        if(search == None):
            query_body = build_elastic_recommender_query(
                limit,
                "",
                "",
                "",
                "",
                lang,
                10,
                0,
                second_try=True # this bad boy is needed until we add constrains 

            )
        else:
            query_body = build_elastic_search_query(
                limit,
                search,
                "",
                "",
                "",
                "",
                lang,
                10,
                0,
                second_try=True # this bad boy is needed until we add constrains 
            )
        res = es.search(index=ES_ZINDEX, body=query_body)
        hit_list = res["hits"].get("hits")
        return _to_articles_from_ES_hits(hit_list)
    if(search != None):
        print(f"Cannot search with No limit, returning all articles for {language}\n")
    return Article.find_by_language(lang)

    
def build_candidate_pool_for_user(user_id: int) -> 'list[candidate]':
    '''Returns a list of articles for a user with whatever constraints they have'''
    u = User.find_by_id(user_id)
    count = len(Article.find_by_language(u.learned_language))
    return article_recommendations_for_user(u,count)


def initial_candidate_pool() -> 'list[Article]':
    query = (
        Article.query
         .filter_by(broken=0)
         .join(UserReadingSession, Article.id == UserReadingSession.article_id)
         .distinct()
         .all())

    return query



def find_articles_like(recommended_articles_ids: 'list[int]', limit: int, article_age: int, language_id: int) -> 'list[Article]':
    es = Elasticsearch(ES_CONN_STRING)
    fields = ["language", "content", "title"]
    language = Language.find_by_id(language_id)
    like_documents = [
        {"_index": ES_ZINDEX, "_id": str(doc_id) } for doc_id in recommended_articles_ids
    ]

    cutoff_date = datetime.now() - timedelta(days=article_age)

    mlt_query = {
        "query": {
            "bool": {
                "must": [
                    {'match': {'language': language.name}}
                ],
                "should": {  
                    "more_like_this": {
                        "fields": fields,
                        "like": like_documents,
                        "min_term_freq": 2, 
                        "max_query_terms": 25, 
                        "min_doc_freq": 5, 
                        "min_word_length" : 3
                    }
                },
                "filter": {
                    "range": {
                        "published_time": {
                            "gte": cutoff_date.strftime('%Y-%m-%dT%H:%M:%S'),
                            "lte": "now"
                        }
                    }
                }
            }
        },
        "sort": [{"published_time": {"order": "desc"}}]
    }

    # Execute the query
    res = es.search(index=ES_ZINDEX, body=mlt_query, size=limit)
    return _to_articles_from_ES_hits(res["hits"]["hits"])