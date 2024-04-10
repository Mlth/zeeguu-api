from .mapper import Mapper
from .recommender_system import (
    RecommenderSystem,
)
class RecommenderSystemSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.mapper = Mapper()
            self.num_users = self.mapper.num_users
            self.num_items = self.mapper.num_articles
            self._initialized = True

    def get_recommender_system(self):
        return RecommenderSystem(None, mapper=self.mapper, num_users=self.num_users, num_items=self.num_items)
