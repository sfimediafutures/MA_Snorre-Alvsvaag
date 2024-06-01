from typing import List, Dict
from rec.types.types import Recommendation, RecommendedItem 
import logging

class Reranker:
    def __init__(self, Bridges, CF, logger) -> None:
        self.logger = logger
        self.logger.name = "reranker"
        self.Bridges = Bridges
        self.CF = CF
        self.missing_bridge_count = 0
        self.missing_cf_count = 0
        self.not_enough_bridge_count = 0
        self.not_enough_cf_count = 0

    def recommend(self, userId, item_id, N=5, w1=0.5, w2=0.5, K=5):
        """
        Recommends a list of items for a given user.

        Parameters:
        - userId (int): The ID of the user for whom recommendations are generated.
        - N (int): The number of initial recommendations to retrieve.
        - w1 (float): The weight for the collaborative filtering score.
        - w2 (float): The weight for the bridge score.
        - method (str): The method used to calculate the recommendation scores.
        - K (int): The number of items to consider for reranking.

        Returns:
        - recommended_items (Recommendation): A list of recommended items for the user.
        """
        cf_recs, bridges = self._get_recs(userId, item_id, N, K)
        if cf_recs is None or bridges is None:
            return None
        recommended_items = self._rerank(userId, item_id, cf_recs, bridges, w1, w2, N)
        return recommended_items
    
    def _get_recs(self, user_id, item_id, N, K):
        # WE CONSIDER K
        cf_recs = self.CF.recommend_standard(user_id, N=K)
        if cf_recs is None:
            self.missing_cf_count += 1
            return None, None
        
        # WE CONSIDER K
        bridges = self.Bridges.recommend_standard(item_id, N=K)
        if bridges is None:
            self.missing_bridge_count += 1
            return None, None 

        if len(cf_recs.items) < K:
            self.not_enough_cf_count += 1
            return None, None
        if len(bridges.items) < K:
            self.not_enough_bridge_count += 1
            return None, None
        return cf_recs, bridges
    
    def _rerank(self, user_id: str, item_id: str, cf_recs: Recommendation, bridges: Recommendation, w1: float, w2: float, N: int) -> Recommendation:
        recs = Recommendation(user_id=user_id, item_id=item_id, items_map={}, items=[], item_ids=[]) 
        # we perform softmax on both the CF and bridge scores
        cf_recs.softmax_normalize_scores()
        bridges.softmax_normalize_scores()
        overlap = 0
        # Score and add the items to the recs
        for recommended_item in cf_recs.items:
            # Start by scoring the CF item:
            recommended_item.score = recommended_item.score * w1
            # check if the item is in the bridge recommendations
            bridge_item = bridges.items_map.get(recommended_item.item_id, None)
            if bridge_item is not None:
                overlap += 1
                # If the item is in the bridge recommendations, we add the bridge score to the CF score, weighted by w2
                recommended_item.score = (recommended_item.score) + (w2 * bridge_item.score)
                # We add it to the item map of our reranked recommendation, so that we are able to keep track of the bridge items,
                # when we add the bridge items after, we dont want to add the same items twice
                recs.items_map[recommended_item.item_id] = RecommendedItem(recommended_item.item_id, recommended_item.score, "RERANK")
                # We add the item to the list of items
                recs.item_ids.append(recommended_item.item_id)
                recs.items.append(RecommendedItem(recommended_item.item_id, recommended_item.score, "RERANK"))
            else:
                # If the item is not in the bridge recommendations, we add it to the list of items, but not to the item map. 
                recs.items.append(recommended_item)
                recs.item_ids.append(recommended_item.item_id)
    

        # Add the bridge items that were not in the CF recommendations
        for bridge_item in bridges.items:
            #  We only add the bridge items that were not in the CF recommendations
            if bridge_item.item_id not in recs.items_map:
                # Score the bridge items with weight w2
                bridge_item.score = bridge_item.score * w2
                recs.items.append(bridge_item)
        
        # Sort the items by score and take N recommendations
        recs.items = sorted(recs.items, key=lambda x: x.score, reverse=True)
        if len(recs.items) > N:
            recs.items = recs.items[:N]
            recs.item_ids = [item.item_id for item in recs.items]
            return recs
        self.logger.error("Reranked recommendations less than K.")
        return None