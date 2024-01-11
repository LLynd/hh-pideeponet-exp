import tensorflow as tf 
from tensorflow import keras


class BiasLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(4),
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)
    def call(self, inputs):
        return inputs + self.bias


class Slice(keras.layers.Layer):
    def __init__(self, begin, size,**kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config

    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.size)


class Split(keras.layers.Layer):
    def __init__(self, num_or_size_splits, axis,**kwargs):
        super(Split, self).__init__(**kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_or_size_splits': self.num_or_size_splits,
            'axis': self.axis,
        })
        return config

    def call(self, inputs):
        return tf.split(inputs, self.num_or_size_splits, self.axis)


def create_model(mean, var, verbose=False):
    """Definition of a DeepONet with fully connected branch and trunk layers.

    Args:
    ----
    mean: dictionary, mean values of the inputs
    var: dictionary, variance values of the inputs
    verbose: boolean, indicate whether to show the model summary

    Outputs:
    --------
    model: the DeepONet model
    """

    # Branch net
    branch_input = tf.keras.Input(shape=(len(mean['forcing'])), name="forcing")
    branch = tf.keras.layers.Normalization(mean=mean['forcing'], variance=var['forcing'])(branch_input)
    for i in range(4):
        branch = tf.keras.layers.Dense(200, activation="tanh", name='branch_'+str(i))(branch)

    # Trunk net
    trunk_input = tf.keras.Input(shape=(len(mean['time'])), name="time")
    trunk = tf.keras.layers.Normalization(mean=mean['time'], variance=var['time'])(trunk_input)
    for i in range(4):
        trunk = tf.keras.layers.Dense(200, activation="tanh", name='trunk_'+str(i))(trunk)

    # Compute the dot product between branch and trunk net
    v_neurons_branch, m_neurons_branch, h_neurons_branch, n_neurons_branch = Split(4, axis=1)(branch)
    v_neurons_trunk, m_neurons_trunk, h_neurons_trunk, n_neurons_trunk = Split(4, axis=1)(trunk)

    dot_product_v = tf.keras.layers.Dot(axes=-1)([v_neurons_branch, v_neurons_trunk])
    dot_product_m = tf.keras.layers.Dot(axes=-1)([m_neurons_branch, m_neurons_trunk])
    dot_product_h = tf.keras.layers.Dot(axes=-1)([h_neurons_branch, h_neurons_trunk])
    dot_product_n = tf.keras.layers.Dot(axes=-1)([n_neurons_branch, n_neurons_trunk])
    
    # Add the bias
    output = [dot_product_v, dot_product_m, dot_product_h, dot_product_n]
    output = BiasLayer()(output)

    # Create the model
    model = tf.keras.models.Model(inputs=[branch_input, trunk_input], outputs=output)

    if verbose:
        model.summary()

    return model