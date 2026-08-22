import numpy as np

# Re-run quick analysis of where the headroom comes from: which configs won which query classes?
from sklearn.metrics import ndcg_score
import json

# Let's inspect the distribution of optimal choices per query class
print("Analyzing oracle config wins per query class...")
