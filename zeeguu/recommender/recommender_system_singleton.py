from .mapper import Mapper
from .recommender_system import (
    RecommenderSystem,
)
class RecommenderSystemSingleton:
    _instance = None


    def get_recommender(self):
        if self._instance is None:
            self.mapper = Mapper()
            self.num_users = self.mapper.num_users
            self.num_items = self.mapper.num_articles
            self._initialized = True
            self._instance = RecommenderSystem(
                sessions=None,
                num_users=self.num_users,
                num_items=self.num_items,
                mapper=self.mapper,
            )
        
        return self._instance

    
