"""

 Recommender that uses ElasticSearch instead of mysql for searching.
 Based on mixed recommender.
 Still uses MySQL to find relations between the user and things such as:
   - topics, language and user subscriptions.

"""
from collections import namedtuple
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, SF
import concurrent.futures
import time
import os

from zeeguu.core.model import (
    Article,
    TopicFilter,
    TopicSubscription,
    SearchFilter,
    SearchSubscription,
    UserArticle,
    Language,
)

from zeeguu.core.elastic.elastic_query_builder import (
    build_elastic_recommender_query,
    build_elastic_search_query,
)
from zeeguu.core.util.timer_logging_decorator import time_this
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX


def _prepare_user_constraints(user):
    language = user.learned_language

    # 0. Ensure appropriate difficulty
    declared_level_min, declared_level_max = user.levels_for(language)
    lower_bounds = declared_level_min * 10
    upper_bounds = declared_level_max * 10

    # 1. Unwanted user topics
    # ==============================
    user_search_filters = SearchFilter.all_for_user(user)
    unwanted_user_topics = []
    for user_search_filter in user_search_filters:
        unwanted_user_topics.append(user_search_filter.search.keywords)
    print(f"keywords to exclude: {unwanted_user_topics}")

    # 2. Topics to exclude / filter out
    # =================================
    excluded_topics = TopicFilter.all_for_user(user)
    topics_to_exclude = [each.topic.title for each in excluded_topics]
    print(f"topics to exclude: {topics_to_exclude}")

    # 3. Topics subscribed, and thus to include
    # =========================================
    topic_subscriptions = TopicSubscription.all_for_user(user)
    topics_to_include = [
        subscription.topic.title
        for subscription in TopicSubscription.all_for_user(user)
    ]
    print(f"topics to include: {topic_subscriptions}")

    # 4. Wanted user topics
    # =========================================
    user_subscriptions = SearchSubscription.all_for_user(user)

    wanted_user_topics = []
    for sub in user_subscriptions:
        wanted_user_topics.append(sub.search.keywords)
    print(f"keywords to include: {wanted_user_topics}")

    return (
        language,
        upper_bounds,
        lower_bounds,
        _list_to_string(topics_to_include),
        _list_to_string(topics_to_exclude),
        _list_to_string(wanted_user_topics),
        _list_to_string(unwanted_user_topics),
    )


def article_recommendations_for_user(
        user,
        count,
        es_scale="30d",
        es_offset="1d",
        es_decay=0.6,
        es_weight=4.2,
):
    """

            Retrieve :param count articles which are equally distributed
            over all the feeds to which the :param user is registered to.

            Fails if no language is selected.

    :return:

    """

    final_article_mix = []

    (
        language,
        upper_bounds,
        lower_bounds,
        topics_to_include,
        topics_to_exclude,
        wanted_user_topics,
        unwanted_user_topics,
    ) = _prepare_user_constraints(user)

    # build the query using elastic_query_builder
    query_body = build_elastic_recommender_query(
        count,
        topics_to_include,
        topics_to_exclude,
        wanted_user_topics,
        unwanted_user_topics,
        language,
        upper_bounds,
        lower_bounds,
        es_scale,
        es_offset,
        es_decay,
        es_weight,
    )

    es = Elasticsearch(ES_CONN_STRING)

    #big count uses scroll api instead to chunk it in to manageable parts
    if count > 1000:
        return article_recommendations_for_big_queries(query_body, es)

    res = es.search(index=ES_ZINDEX, body=query_body)
    
    hit_list = res["hits"].get("hits")
    final_article_mix.extend(_to_articles_from_ES_hits(hit_list))

    if len(final_article_mix) == 0:
        # build the query using elastic_query_builder
        query_body = build_elastic_recommender_query(
            count,
            topics_to_include,
            topics_to_exclude,
            wanted_user_topics,
            unwanted_user_topics,
            language,
            upper_bounds,
            lower_bounds,
            es_scale,
            es_decay,
            es_weight,
            second_try=True,
        )
        res = es.search(index=ES_ZINDEX, body=query_body)

    hit_list = res["hits"].get("hits")
    final_article_mix.extend(_to_articles_from_ES_hits(hit_list))

    articles = [a for a in final_article_mix if a is not None and not a.broken]

    sorted_articles = sorted(articles, key=lambda x: x.published_time, reverse=True)

    return sorted_articles


@time_this
def article_search_for_user(
        user,
        count,
        search_terms,
        es_scale="3d",
        es_decay=0.8,
        es_weight=4.2,
):
    final_article_mix = []

    (
        language,
        upper_bounds,
        lower_bounds,
        topics_to_include,
        topics_to_exclude,
        wanted_user_topics,
        unwanted_user_topics,
    ) = _prepare_user_constraints(user)

    # build the query using elastic_query_builder
    query_body = build_elastic_search_query(
        count,
        search_terms,
        topics_to_include,
        topics_to_exclude,
        wanted_user_topics,
        unwanted_user_topics,
        language,
        upper_bounds,
        lower_bounds,
        es_scale,
        es_decay,
        es_weight,
    )

    es = Elasticsearch(ES_CONN_STRING)
    res = es.search(index=ES_ZINDEX, body=query_body)

    hit_list = res["hits"].get("hits")
    final_article_mix.extend(_to_articles_from_ES_hits(hit_list))

    if len(final_article_mix) == 0:
        # build the query using elastic_query_builder
        query_body = build_elastic_search_query(
            count,
            search_terms,
            topics_to_include,
            topics_to_exclude,
            wanted_user_topics,
            unwanted_user_topics,
            language,
            upper_bounds,
            lower_bounds,
            es_scale,
            es_decay,
            es_weight,
            second_try=True,
        )
        res = es.search(index=ES_ZINDEX, body=query_body)

        hit_list = res["hits"].get("hits")
        final_article_mix.extend(_to_articles_from_ES_hits(hit_list))

    return [a for a in final_article_mix if a is not None and not a.broken]


def topic_filter_for_user(user,
                          count,
                          newer_than,
                          media_type,
                          max_duration,
                          min_duration,
                          difficulty_level,
                          topic):
    es = Elasticsearch(ES_CONN_STRING)

    s = Search().query(Q("term", language=user.learned_language.code()))

    if newer_than:
        s = s.filter("range", published_time={"gte": f"now-{newer_than}d/d"})

    AVERAGE_WORDS_PER_MINUTE = 70

    if max_duration:
        s = s.filter("range", word_count={"lte": int(max_duration) * AVERAGE_WORDS_PER_MINUTE})

    if min_duration:
        s = s.filter("range", word_count={"gte": int(min_duration) * AVERAGE_WORDS_PER_MINUTE})

    if media_type:
        if media_type == "video":
            s = s.filter("term", video=1)
        else:
            s = s.filter("term", video=0)

    if topic != None and topic != "all":
        s = s.filter("match", topics=topic.lower())

    if difficulty_level:
        lower_bounds, upper_bounds = _difficuty_level_bounds()
        s = s.filter("range", fk_difficulty={"gte": lower_bounds,
                                             "lte": upper_bounds})

    query = s.query

    query_with_size = {"size": count,
                       "query": query.to_dict(),
                       "sort": [{"published_time": "desc"}]}

    res = es.search(index=ES_ZINDEX, body=query_with_size)

    hit_list = res["hits"].get("hits")

    final_article_mix = _to_articles_from_ES_hits(hit_list)

    return [a for a in final_article_mix if a is not None and not a.broken]


def _list_to_string(input_list):
    return " ".join([each for each in input_list]) or ""


def _to_articles_from_ES_hits(hits):
    articles = []
    for hit in hits:
        articles.append(Article.find_by_id(hit.get("_id")))
    return articles


def _difficuty_level_bounds(level):
    lower_bounds = 1
    upper_bounds = 10

    if level == "easy":
        upper_bounds = 5
    elif level == "challenging":
        lower_bounds = 5
    else:
        lower_bounds = 4
        upper_bounds = 8
    return lower_bounds, upper_bounds


candidate = namedtuple('candidate', ['article_id', 'score'])

#helper function only called by article_recommendations_for_big_queries
def helper(id,query_body,es,thread_amount) -> list[candidate]:
    
    query_body_with_slice = {
    "slice": {
        "id": id,  # Set the slice id
        "max": thread_amount   # Set the maximum number of slices
    },
    **query_body  # Merge the original query body with the slice parameter
    }
    res = es.search(
        index=ES_ZINDEX,
        body=query_body_with_slice,
        scroll='2s', #keep alive time 
        size=256, # number of hits per slice
    )
    scroll_id = res["_scroll_id"]
    count = res['hits']['total']['value']
    mix = [None] * count
    hits = res['hits']['hits']
    ptr = 0 
    for _, hit in enumerate(hits):
        mix[ptr] = candidate(hit['_id'], hit['_score'])
        ptr += 1 
    while len(hits) > 0:
        try:
            scan_results = es.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = scan_results['_scroll_id']
            hits = scan_results['hits']['hits']
            for _, hit in enumerate(hits):
                mix[ptr] = candidate(hit['_id'], hit['_score'])
                ptr += 1 
        except Exception as e:
            print(f"Error occurred during scroll: {e}")
            break
    
    es.clear_scroll(scroll_id=scroll_id)

    return mix

def article_recommendations_for_big_queries(query_body, es) -> list[candidate]:
    thread_amount = len(os.sched_getaffinity(0)) #current amount of available cpus in sys that python can access
    final_candidate_mix = []
    thread_ids=range(0, thread_amount)

    #allocate a thread for each slice in elastic and execute es.search for them in parrallel 
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(helper, id,query_body, es,thread_amount) for id in thread_ids]
    
    for c in [f.result() for f in futures]:
        final_candidate_mix.extend(c)

    #remove any articles that are broken and sort by score ascending
    article_ids = [c.article_id for c in final_candidate_mix]
    articles = Article.query.filter_by(broken=1).filter(Article.id.in_(article_ids)).all()
    article_ids_to_remove = {a.id for a in articles} 
    candidates_with_scores = [
    candidate(article_id=c.article_id, score=c.score) for c in final_candidate_mix if int(c.article_id) not in article_ids_to_remove
    ]
    sorted_candidates = sorted(candidates_with_scores, key=lambda candidate: candidate.score)
    
    return sorted_candidates
