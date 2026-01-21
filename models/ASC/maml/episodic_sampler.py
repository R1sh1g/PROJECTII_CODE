# src/episodic_sampler.py

import random
from typing import Dict, List, Tuple

def sample_episode(index: Dict, aspect: str, n_way: int, k_shot: int, q_query: int):
    """
    Episode for one aspect:
      support: N-way * K-shot
      query:   N-way * Q
    """
    support = []
    query = []
    for lab in range(n_way):
        pool = index[aspect][lab]
        chosen = random.sample(pool, k_shot + q_query)
        support.extend(chosen[:k_shot])
        query.extend(chosen[k_shot:])
    random.shuffle(support)
    random.shuffle(query)
    return support, query
