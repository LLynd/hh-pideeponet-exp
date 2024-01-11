import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['figure.dpi'] = 100
plt.subplots_adjust(top = 1, bottom = 0.5, wspace=2, hspace=4)

def plot_history():
    fig, ax = plt.subplots(1, 4)
    ax[0].plot(range(n_epochs), loss_tracker.loss_history['IC_loss'])
    ax[1].plot(range(n_epochs), loss_tracker.loss_history['ODE_loss'])
    ax[2].plot(range(n_epochs), loss_tracker.loss_history['Data_loss'])
    ax[3].plot(range(n_epochs), val_loss_hist)
    ax[0].set_title('IC Loss')
    ax[1].set_title('ODE Loss')
    ax[2].set_title('Data Loss')
    ax[3].set_title('Val Loss')
    for axs in ax:
        axs.set_yscale('log')
    
    
#@title Sample input/output of the model with true response
def plot_model_response():
    from copy import copy
    v_single, m_single, n_single, h_single, i_single, t_single = calculate_HH_model_response(signal.square, 100)
    print(i_single.shape)
    print(t_single.shape)
    print(X_train[:, 1:-1].shape)
    print(X_train[:, :1].shape)
    data = np.zeros((1000, 1001))
    for i in range(1000):
    data[i, :-1] = i_single[:]
    data[i, -1] = t_single[i]

    print(data.shape)
    print(data[:, :-1].shape)
    print(data[:, -1].shape)
    preds = PI_DeepONet.predict({"forcing": tf.convert_to_tensor(data[:, :-1], dtype=tf.float32), "time": tf.convert_to_tensor(np.expand_dims(data[:, -1], 1), dtype=tf.float32)})
    preds = np.array(preds)
    print(preds.shape)

    v_pred = preds[0, :, :].flatten()
    m_pred = preds[1, :, :].flatten()
    h_pred = preds[2, :, :].flatten()
    n_pred = preds[3, :, :].flatten()

    plt.subplot(5, 1, 1)
    plt.plot(t_single, i_single)
    plt.title('input I, single')
    plt.subplot(5, 1, 2)
    plt.plot(t_single, v_single)
    plt.plot(t_single, v_pred*100)
    plt.title('output V, single')
    plt.subplot(5, 1, 3)
    plt.plot(t_single, m_single)
    plt.plot(t_single, m_pred)
    plt.title('output m, single')
    plt.subplot(5, 1, 4)
    plt.plot(t_single, h_single)
    plt.plot(t_single, h_pred)
    plt.title('output h, single')
    plt.subplot(5, 1, 5)
    plt.plot(t_single, n_single)
    plt.plot(t_single, n_pred)
    plt.title('output n, single')

def test_model():
    v_single, m_single, n_single, h_single, i_single, t_single = calculate_HH_model_response(signal.square, 100)

    plt.subplot(3, 1, 1)
    plt.plot(t_single, i_single)
    plt.title('Przebieg impulsu stymulującego I')
    plt.xlabel('Czas (ms)')
    plt.ylabel('Natężenie (nA)')
    plt.subplot(3, 1, 2)
    plt.plot(t_single, v_single)
    plt.title('Odpowiedź neuronu HH, napięcie V')
    plt.xlabel('Czas (ms)')
    plt.ylabel('Napięcie (mV)')
    plt.subplot(3, 1, 3)
    plt.plot(t_single, m_single, label='m')
    plt.plot(t_single, n_single, label='n')
    plt.plot(t_single, h_single, label='h')
    plt.title('Odpowiedź neuronu HH, zmienne m, n, h')
    plt.legend()
    plt.tight_layout()
    plt.savefig("single-neuron-simulation.png")