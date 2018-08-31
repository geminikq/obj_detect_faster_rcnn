import tensorflow as tf
import numpy as np
from model.rpn.anchor_gen import generate_anchors, generate_shifted_anchors
from model.rpn.anchor_tensor_gen import generate_shifted_anchors_tensor


base_anchor_size = 16
anchor_ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]
feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 18])
fshape = tf.shape(feature_map)
height_t = fshape[1]
width_t = fshape[2]
height = 3
width = 2
fmap_test = \
    [
        [
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            ]

        ]
    ]
fmap_shape = np.array(fmap_test).shape


anchor_py = generate_anchors(base_anchor_size, anchor_ratios, anchor_scales)
shifted_anchor_py = generate_shifted_anchors(height, width, anchor_py, [16])

anchor_tensor = tf.convert_to_tensor(anchor_py, tf.float32)
all_anchors = generate_shifted_anchors_tensor(height, width, anchor_tensor, [16])

print(shifted_anchor_py)

sess = tf.Session()
anchor_t = sess.run(all_anchors, feed_dict={feature_map: fmap_test})

print(anchor_t)

print('done')
