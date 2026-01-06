from fastapi import FastAPI
import numpy as np
from config import RecommendRequest, UserAction, BATCH_SIZE, EPS
import tensorflow as tf

from training import recommend_for_user
from utils import update_user_tags, reward, Action
from replay_buffer.replaybuffer import *
from training import train_from_replay, ITEMS, ITEM_RATING_NORM


app = FastAPI()   

# Load replay buffer nếu file tồn tại
try:
    replay_buffer = load_replay_buffer("replay_buffer/" + REPLAY_BUFFER_FILE)
except FileNotFoundError:
    replay_buffer = ReplayBuffer(capacity=10000)
    print("Không có file cũ → tạo buffer mới.")


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    # In ra để kiểm tra
    print("Received user_id:", req.user_id)
    print("Received user_bias:", req.user_bias)

    # Chuyển về mảng chỉ có các giá trị bias
    user_tags = np.array(list(req.user_bias.values()), dtype=np.float32)
    topk, qvals = recommend_for_user(req.user_id, np.array(user_tags), EPS)

    return {
        "recommended_items": topk,
        "scores": qvals
    }
    

@app.put("/push_replay_buffer")
async def push_replay_buffer(req: UserAction):
    user_tags = np.array(req.user_tags, dtype=np.float32)   # user_tags là list float
    dish_tags = np.array(req.dish_tags, dtype=np.float32)   # dish_tags là list 0/1
    action = req.action
    item_id = req.item_id

    next_state = update_user_tags(user_tags, dish_tags, action)
    r = reward(action)

    # tạo transition
    transition = Transition(
        user_id=req.user_id,
        user_tags=user_tags,
        item_id=item_id,
        reward=r,
        next_user_tags=next_state
    )

    # đẩy vào replay buffer
    replay_buffer.push(*transition)
    save_replay_buffer(replay_buffer, "replay_buffer/" + REPLAY_BUFFER_FILE)

    if len(replay_buffer) >= BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        loss = train_from_replay(batch)
        print("Model trained, loss =", loss)


    return {"status": "ok", "buffer_size": len(replay_buffer)}



@app.post("/reload_model")
async def reload_model():
    """Endpoint để reload model khi có thay đổi số tags"""
    from training import policy_net, target_net, ACTUAL_NUM_TAGS
    
    # Build lại model với số tags mới
    _dummy_user_id = tf.constant([[0]])
    _dummy_tags = tf.constant(np.zeros((1, ACTUAL_NUM_TAGS), dtype=np.float32))
    _dummy_items = tf.constant([ITEMS.tolist()], dtype=tf.int32)
    
    _ = policy_net.call_all_q(_dummy_user_id, _dummy_tags, _dummy_items, 
                             item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32))
    _ = target_net.call_all_q(_dummy_user_id, _dummy_tags, _dummy_items, 
                             item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32))
    
    return {"status": "Model reloaded", "num_tags": ACTUAL_NUM_TAGS}


# @app.post("/recommend")
# def recommend(user: UserData):
#     topk, qvals = recommend_for_user(user.user_id, np.array(user.user_tags))
#     return {"recommended_items": topk, "scores": qvals}



