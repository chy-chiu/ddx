import numpy as np
from typing import Dict

def sample_from_distribution(distribution: Dict[int, float]):
    
    k, v = zip(*distribution.items())
    
    return np.random.choice(k, p=v)
