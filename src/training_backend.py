import tensorflow as tf
from tensorflow import keras
from collections import defaultdict

from residual import HH_residual_calculator


@tf.function
def train_step(X, X_init, IC_weight, ODE_weight, model, y_true):
    """Calculate gradients of the total loss with respect to network model parameters.

    Args:
    ----
    X: training dataset for evaluating ODE residuals
    X_init: training dataset for evaluating initial conditions
    IC_weight: weight for initial condition loss
    ODE_weight: weight for ODE loss
    model: DeepONet model

    Outputs:
    --------
    ODE_loss: calculated ODE loss
    IC_loss: calculated initial condition loss
    total_loss: weighted sum of ODE loss and initial condition loss
    gradients: gradients of the total loss with respect to network model parameters.
    """
    IC_weight = tf.constant(2.0, dtype=tf.float32)
    v_weight = tf.constant(1.0, dtype=tf.float32)
    m_weight = tf.constant(1.0, dtype=tf.float32)
    h_weight = tf.constant(1.0, dtype=tf.float32)
    n_weight = tf.constant(1.0, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_weights)

        # Initial condition prediction
        y_pred = model({"forcing": X[:, 1:-1], "time": X[:, :1]})
        y_pred_IC = model({"forcing": X_init[:, 1:-1], "time": X_init[:, :1]})

        # Equation residual
        v_residual, m_residual, h_residual, n_residual = HH_residual_calculator(t=X[:, :1], u=X[:, 1:-1], u_t=X[:, -1:], model=model)

        # Calculate loss
        data_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_true, y_pred))
        IC_loss = tf.reduce_mean(keras.losses.mean_squared_error(0, y_pred_IC))
        v_loss = tf.reduce_mean(tf.square(v_residual))
        m_loss = tf.reduce_mean(tf.square(m_residual))
        h_loss = tf.reduce_mean(tf.square(h_residual))
        n_loss = tf.reduce_mean(tf.square(n_residual))

        # Total loss
        total_loss = IC_loss*IC_weight + v_loss*v_weight + m_loss*m_weight + h_loss*h_weight + n_loss*n_weight + data_loss*IC_weight

    gradients = tape.gradient(total_loss, model.trainable_variables)

    return v_loss+m_loss+h_loss+n_loss, IC_loss, data_loss, total_loss, gradients


class LossTracking:

    def __init__(self):
        self.mean_total_loss = keras.metrics.Mean()
        self.mean_IC_loss = keras.metrics.Mean()
        self.mean_ODE_loss = keras.metrics.Mean()
        self.mean_Data_loss = keras.metrics.Mean()
        self.loss_history = defaultdict(list)

    def update(self, total_loss, IC_loss, ODE_loss, data_loss):
        self.mean_total_loss(total_loss)
        self.mean_IC_loss(IC_loss)
        self.mean_ODE_loss(ODE_loss)
        self.mean_Data_loss(data_loss)

    def reset(self):
        self.mean_total_loss.reset_states()
        self.mean_IC_loss.reset_states()
        self.mean_ODE_loss.reset_states()
        self.mean_Data_loss.reset_states()

    def print(self):
        print(f"IC={self.mean_IC_loss.result().numpy():.4e}, \
              ODE={self.mean_ODE_loss.result().numpy():.4e}, \
              Data={self.mean_Data_loss.result().numpy():.4e}, \
              total_loss={self.mean_total_loss.result().numpy():.4e}")

    def history(self):
        self.loss_history['total_loss'].append(self.mean_total_loss.result().numpy())
        self.loss_history['IC_loss'].append(self.mean_IC_loss.result().numpy())
        self.loss_history['ODE_loss'].append(self.mean_ODE_loss.result().numpy())
        self.loss_history['Data_loss'].append(self.mean_Data_loss.result().numpy())