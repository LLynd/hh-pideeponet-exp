import tensorflow as tf


non_tf_constant_dict = {
    'Cm': 0.000001 * 2000 * 0.000001 * 0.000001 / 0.0001,
    'g_na': 100 * 0.001 * 2000 * 0.000001 * 0.000001 / 0.0001,
    'g_k': 30 * 0.001 * 2000 * 0.000001 * 0.000001 / 0.0001,
    'g_l': 0.00005 * 2000 * 0.000001 * 0.000001 / 0.0001,
    'v_na': 50 * 0.001,
    'v_k': -90 * 0.001,
    'v_l':  -65 * 0.001,
    'w': 40 * 0.000000001,
    'e_synap': -75 * 0.001,
}

constant_dict = {k: tf.constant(v, dtype=tf.float32) for k, v in non_tf_constant_dict.items()}