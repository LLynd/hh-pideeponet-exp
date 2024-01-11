import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config_dc import config
from src.constants import constant_dict
from src.models import create_model
from src.training_backend import train_step, LossTracking
from src.residual import HH_residual_calculator


def train(X_train, y_train, X_val, y_val, cfg, X_test=None, y_test=None):
    # Get parameters from config
    ini_batch_size = cfg.ini_batch_size
    col_batch_size = cfg.col_batch_size
    n_epochs = cfg.n_epochs
    IC_weight = cfg.IC_weight
    ODE_weight = cfg.ODE_weight
    npy_res_path = cfg.npy_res_path

    # Save config
    with open(os.path.join('..', cfg.cfg_path), 'w', encoding="utf8") as fp:
        json.dump(cfg.to_dict() , fp)

    # Create dataset object (initial conditions)
    X_train_ini = tf.convert_to_tensor(X_train[X_train[:, 0]==0], dtype=tf.float32)
    ini_ds = tf.data.Dataset.from_tensor_slices((X_train_ini))
    ini_ds = ini_ds.batch(ini_batch_size) #.shuffle(5000)


    # Create dataset object (collocation points)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train))
    train_ds = train_ds.batch(col_batch_size) #.shuffle(100000)

    y_true = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_true = tf.data.Dataset.from_tensor_slices((y_true))
    y_true = y_true.batch(col_batch_size)

    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    if X_test is not None:
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    if y_test is not None:
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    # Scaling
    mean = {
        'forcing': np.mean(X_train[:, 1:-1], axis=0),
        'time': np.mean(X_train[:, :1], axis=0)
    }

    var = {
        'forcing': np.var(X_train[:, 1:-1], axis=0),
        'time': np.var(X_train[:, :1], axis=0)
    }
    
    # Set up training configurations
    loss_tracker = LossTracking()
    val_loss_hist = []

    # Set up optimizer
    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    # Instantiate the PINN model
    PI_DeepONet= create_model(mean, var, False)
    PI_DeepONet.compile(optimizer=optimizer)

    model_config = PI_DeepONet.get_config() # Returns pretty much every information about your model
    tf.keras.utils.plot_model(PI_DeepONet, os.path.join('result', 'PI_DeepONet.png'))

    # Configure callback
    _callbacks = [keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10),
                tf.keras.callbacks.ModelCheckpoint('NN_model.h5', monitor='val_loss', save_best_only=True)]
    callbacks = tf.keras.callbacks.CallbackList(
                    _callbacks, add_history=False, model=PI_DeepONet)

    # Start training process
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}:")
        for X_init, X, yt in zip(ini_ds, train_ds, y_true):
            # Calculate gradients
            ODE_loss, IC_loss, data_loss, total_loss, gradients = train_step(X, X_init,
                                                                            IC_weight,
                                                                            ODE_weight,
                                                                            PI_DeepONet,
                                                                            yt)
            # Gradient descent
            PI_DeepONet.optimizer.apply_gradients(zip(gradients, PI_DeepONet.trainable_variables))

            # Loss tracking
            loss_tracker.update(total_loss, IC_loss, ODE_loss, data_loss)

        # Loss summary
        loss_tracker.history()
        loss_tracker.print()
        loss_tracker.reset()

        #Validation
        val_v_residual, val_m_residual, val_h_residual, val_n_residual = HH_residual_calculator(X_val[:, :1], X_val[:, 1:-1], X_val[:, -1:], PI_DeepONet)
        val_v_ode = tf.cast(tf.reduce_mean(tf.square(val_v_residual)), tf.float32)
        val_m_ode = tf.cast(tf.reduce_mean(tf.square(val_m_residual)), tf.float32)
        val_h_ode = tf.cast(tf.reduce_mean(tf.square(val_h_residual)), tf.float32)
        val_n_ode = tf.cast(tf.reduce_mean(tf.square(val_n_residual)), tf.float32)

        X_val_ini = X_val[X_val[:, 0]==0]
        pred_ini_valid = PI_DeepONet.predict({"forcing": X_val_ini[:, 1:-1], "time": X_val_ini[:, :1]}, batch_size=12800)
        val_IC = tf.reduce_mean(keras.losses.mean_squared_error(0, pred_ini_valid))
        pred_valid = PI_DeepONet.predict({"forcing": X_val[:, 1:-1], "time": X_val[:, :1]}, batch_size=12800)
        val_data = tf.reduce_mean(keras.losses.mean_squared_error(y_val, pred_valid))
        print(f"val_IC: {val_IC.numpy():.4e}, val_v_ODE: {val_v_ode.numpy():.4e}, val_m_ODE: {val_m_ode.numpy():.4e}, val_h_ODE: {val_h_ode.numpy():.4e}, val_n_ODE: {val_n_ode.numpy():.4e}, val_data: {val_data.numpy():.4e}, lr: {PI_DeepONet.optimizer.lr.numpy():.2e}")

        # Callback at the end of epoch
        callbacks.on_epoch_end(epoch, logs={'val_loss': val_IC+val_v_ode+val_m_ode+val_h_ode+val_n_ode+val_data})
        val_loss_hist.append(val_IC+val_v_ode+val_m_ode+val_h_ode+val_n_ode)
