# ========== Replay Buffer ==========
import pickle
import numpy as np
from collections import deque, namedtuple
import random

REPLAY_BUFFER_FILE = "replay_buffer.pkl"

# Transition tuple
Transition = namedtuple('Transition', ('user_id','user_tags','item_id','reward','next_user_tags'))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n):
        return random.sample(self.buf, n)
    
        # batch = random.sample(self.buf, n)
        # user_ids = [t.user_id for t in batch]
        # user_tags = np.array([t.user_tags for t in batch], dtype=np.float32)
        # actions = [t.action for t in batch]
        # rewards = [t.reward for t in batch]
        # next_user_tags = np.array([t.next_user_tags for t in batch], dtype=np.float32)
        
        # return Transition(user_ids, user_tags, actions, rewards, next_user_tags)

    def __len__(self):
        return len(self.buf)


# =======================
# Hàm lưu buffer
def save_replay_buffer(buffer: ReplayBuffer, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"Replay buffer đã được lưu vào {filename}")


# Hàm load buffer
def load_replay_buffer(filename: str) -> ReplayBuffer:
    with open(filename, 'rb') as f:
        buffer = pickle.load(f)
    print(f"Replay buffer đã được load từ {filename}, có {len(buffer)} transition")
    return buffer




    
