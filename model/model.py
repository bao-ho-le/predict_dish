# pyright: reportMissingImports=false
# Dòng trên đầu dùng để tắt cảnh báo không import tensorflow, 
# không hiểu sao những cứ hiện cảnh báo nhưng không lỗi

from tensorflow.keras.layers import Embedding, Dense
import tensorflow as tf
from config import *

# ========== Model ==========
class RecommenderDQN(tf.keras.Model):
    
    def __init__(self, num_users, num_items, user_emb_dim=16, item_emb_dim=16, hidden=128):
        super().__init__()

        self.user_embedding = Embedding(num_users, user_emb_dim, trainable=True)
        self.item_embedding = Embedding(num_items, item_emb_dim, trainable=True)
        
        self.tags_fc = Dense(16, activation='relu', trainable=True)
        self.user_tag_fc = Dense(16, activation='relu', trainable=True)
        self.item_fc = Dense(item_emb_dim, activation='relu', trainable=True)

        self.alpha = tf.Variable(0.5, trainable=True, dtype=tf.float32)
        self.is_built = False


    def build(self, input_shape=None):
            # Build sẽ được gọi tự động khi model chạy lần đầu
            super().build(input_shape)
            self.is_built = True


    # Các hàm có call ở đầu được gọi là forward pass, nó định nghĩa cách truyền dữ liệu 
    # từ đầu vào được xử lý, đi qua các lớp của mạng neuron, và ra output cuối cùng.
    def call_user_repr(self, user_id, user_tags):

        emb = self.user_embedding(user_id) 

        if len(emb.shape) == 3:
            batch_size = tf.shape(emb)[0]
            emb = tf.reshape(emb, [batch_size, -1]) 

        tags_fc = self.tags_fc(user_tags)

        x = tf.concat([emb, tags_fc], axis=1)              
        user_repr = self.user_tag_fc(x)
         
        return user_repr
    

    def call_all_q(self, user_id, user_tags, items, item_rating_norm=None):
        user_repr = self.call_user_repr(user_id, user_tags)  
        item_embs = self.item_embedding(items)    
        item_repr = self.item_fc(item_embs)     
        
        # Sửa phép matmul để có shape [batch_size, num_items]
        user_repr_expanded = tf.expand_dims(user_repr, 1)  # [batch_size, 1, user_emb_dim]
        q = tf.squeeze(tf.matmul(user_repr_expanded, item_repr, transpose_b=True), axis=1)  # [batch_size, num_items]

        if item_rating_norm is not None:
            # Lấy số items thực tế từ tensor items
            num_current_items = tf.shape(items)[1]  # Số items trong batch hiện tại
            
            # Lấy ratings tương ứng với các items hiện có
            # Chuyển items tensor về dạng indices (giả sử ITEMS là [1,2,3,...])
            items_indices = tf.cast(items[0] - 1, tf.int32)  # Trừ 1 vì indices bắt đầu từ 0
            
            # Lấy ratings cho các items cụ thể này
            current_ratings = tf.gather(item_rating_norm, items_indices)  # [num_current_items]
            
            # Xử lý NaN values
            rating_tensor = tf.where(
                tf.math.is_nan(current_ratings),
                0.0,
                current_ratings - 0.5
            )
            
            # Reshape để khớp với q
            rating_tensor = tf.reshape(rating_tensor, [1, num_current_items])  # [1, num_current_items]
            
            q = q + self.alpha * rating_tensor

        return q
    