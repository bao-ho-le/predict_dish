from config import *
from model.model import *
import random, numpy as np
from replay_buffer.replaybuffer import ReplayBuffer, load_replay_buffer
from utils import fetch_item_rating, fetch_items, load_models, save_models
import os


# Lấy điểm trung bình từng món
ITEM_RATINGS = fetch_item_rating()
ITEM_RATING_NORM = (ITEM_RATINGS - ITEM_RATINGS.min()) / (ITEM_RATINGS.max() - ITEM_RATINGS.min())

# Lấy các món ăn
ITEMS = fetch_items()

try:
    from api import replay_buffer as replay
    print(f"✅ Đã sử dụng replay buffer từ API, kích thước: {len(replay)}")
except ImportError as e:
    print(f"⚠️ Không thể import replay buffer từ API: {e}")
    # Fallback: load từ file
    REPLAY_BUFFER_FILE = "replay_buffer/replay_buffer.pkl"
    try:
        replay = load_replay_buffer(REPLAY_BUFFER_FILE)
        print(f"✅ Đã load replay buffer từ file, kích thước: {len(replay)}")
    except FileNotFoundError:
        replay = ReplayBuffer(REPLAY_CAP)
        print("🆕 Tạo replay buffer mới")


# ========== XÁC ĐỊNH SỐ TAGS THỰC TẾ ==========
def get_actual_num_tags():

    print(f"replay.buf length: {len(replay.buf)}")
    
    if len(replay.buf) > 0:
        sample = replay.buf[0]
        print(f"Sample user_tags length: {len(sample.user_tags)}")
        return len(sample.user_tags)
    else:
        print("🚨 REPLAY BUFFER RỖNG - DÙNG SỐ TAGS MẶC ĐỊNH 21")
        return NUM_TAGS


ACTUAL_NUM_TAGS = get_actual_num_tags()
print(f"✅ Số tags thực tế: {ACTUAL_NUM_TAGS}")


# ========== Model Saving & Loading ==========
MODEL_DIR = "model"
POLICY_NET_PATH = os.path.join(MODEL_DIR, "policy_net")
TARGET_NET_PATH = os.path.join(MODEL_DIR, "target_net")

policy_net = RecommenderDQN(NUM_USERS, NUM_ITEMS, USER_EMB_DIM, ITEM_EMB_DIM, HIDDEN)
target_net = RecommenderDQN(NUM_USERS, NUM_ITEMS, USER_EMB_DIM, ITEM_EMB_DIM, HIDDEN)

# Build model trước khi load weights
_dummy_user_id = tf.constant([[0]])
_dummy_tags = tf.constant(np.zeros((1, ACTUAL_NUM_TAGS), dtype=np.float32))
_dummy_items = tf.constant([ITEMS.tolist()], dtype=tf.int32)

_ = policy_net.call_all_q(_dummy_user_id, _dummy_tags, _dummy_items, 
                         item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32))
_ = target_net.call_all_q(_dummy_user_id, _dummy_tags, _dummy_items, 
                         item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32))


# Đồng bộ weights ban đầu
target_net.set_weights(policy_net.get_weights())

# Sau đó mới load models (nếu có)
try:
    load_models(policy_net, target_net, MODEL_DIR, POLICY_NET_PATH, TARGET_NET_PATH)
except Exception as e:
    print(f"⚠️ Cannot load models: {e}. Using initialized models.")
    


optimizer = tf.keras.optimizers.Adam(LR)


# epsilon schedule
# def epsilon_by_step(step):
#     return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)

# helper to select top-k with exploration
def recommend_for_user(user_id, user_tags_vec, eps):
    user_id_tf = tf.constant([[user_id]], dtype=tf.int32)
    user_tags_tf = tf.constant(user_tags_vec.reshape(1,-1), dtype=tf.float32)

    items_tf = tf.constant(ITEMS, dtype=tf.int32) 
    items_tf = tf.reshape(items_tf, [1, -1])

    qvals = policy_net.call_all_q(
        user_id_tf, 
        user_tags_tf, 
        items_tf,
        item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32)
    )
    
    q_np = qvals.numpy().flatten()
    topk_indices = list(np.argsort(-q_np)[:TOP_K])

    # SỬA: Dùng .tolist() để chuyển numpy array sang Python list
    topk_item_ids = ITEMS[topk_indices].tolist()

    if random.random() < eps:
        candidate_indices = list(set(range(len(ITEMS))) - set(topk_indices))
        if len(candidate_indices) >= NUM_EXPLORE:
            explore_indices = random.sample(candidate_indices, NUM_EXPLORE)
            explore_item_ids = ITEMS[explore_indices].tolist()
            topk_item_ids[-NUM_EXPLORE:] = explore_item_ids
    
    # SỬA: Chuyển q_np sang Python list
    q_values_list = q_np.tolist()
    
    return topk_item_ids, q_values_list


# training step
@tf.function
def train_step(batch_user_ids, batch_user_tags, batch_actions, batch_rewards, batch_next_user_tags):

    items_tf = tf.constant(ITEMS, dtype=tf.int32)  
    items_tf = tf.reshape(items_tf, [1, -1])  

    with tf.GradientTape() as tape:
        
        q_all = policy_net.call_all_q(
            batch_user_ids,
            batch_user_tags,  
            items_tf,
            item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32)
        )
        
        idx = tf.stack([tf.range(tf.shape(batch_actions)[0], dtype=tf.int32), batch_actions], axis=1)
        q_pred = tf.gather_nd(q_all, idx) 

        # compute target using target_net
        next_q_all = target_net.call_all_q(
            batch_user_ids, 
            batch_next_user_tags, 
            items_tf, 
            item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32)
        )        
        
        max_next_q = tf.reduce_max(next_q_all, axis=1)  # (B,)
        target = batch_rewards + GAMMA * max_next_q

        loss = tf.reduce_mean(tf.square(target - q_pred))

    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
    return loss


def train_from_replay(batch):
    # batch = list of transitions
    # mỗi transition = (user_id, user_tags, action, reward, next_user_tags)

    batch_user_ids = tf.constant([b[0] for b in batch], dtype=tf.int32)
    batch_user_tags = tf.constant([b[1] for b in batch], dtype=tf.float32)
    batch_items_id = tf.constant([b[2] for b in batch], dtype=tf.int32)
    batch_rewards = tf.constant([b[3] for b in batch], dtype=tf.float32)
    batch_next_user_tags = tf.constant([b[4] for b in batch], dtype=tf.float32)

    loss = train_step(
        batch_user_ids,
        batch_user_tags,
        batch_items_id,
        batch_rewards,
        batch_next_user_tags
    )

    save_models(policy_net, target_net, MODEL_DIR, POLICY_NET_PATH, TARGET_NET_PATH)

    return loss


