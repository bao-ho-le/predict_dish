from enum import Enum
from pydantic import BaseModel
from typing import List, Dict

# ========== Config ==========
NUM_USERS = 100
NUM_ITEMS = 100
NUM_TAGS  = 100
USER_EMB_DIM = 16
ITEM_EMB_DIM = 16
HIDDEN = 128
GAMMA = 0.99
LR = 1e-3
REPLAY_CAP = 20000
TARGET_UPDATE_EVERY = 500  
MAX_STEPS = 20000
EPS = 0.2
TOP_K = 10
NUM_EXPLORE = 2  
# BATCH_SIZE = 64
BATCH_SIZE = 10



# ========== Type ==========
class Action(str, Enum):
    DETAILS = "DETAILS"
    ADD_TO_CART = "ADD_TO_CART"
    ORDER = "ORDER"

action_map = {
    Action.DETAILS: 0,
    Action.ADD_TO_CART: 1,
    Action.ORDER: 2
}


class RecommendRequest(BaseModel):
    user_id: int
    user_bias: Dict[str, float]      
              


class UserAction(BaseModel):
    user_id: int
    user_tags: List[float]
    dish_tags: List[int]
    action: Action
    item_id: int 