import tensorflow as tf

from src.constants import constant_dict


@tf.function
def alfa_n(v):
  return (0.01 * (10 - v)) / tf.math.exp((10-v)/10) - 1

@tf.function
def alfa_h(v):
  return 0.07 * tf.math.exp(-v/20)

@tf.function
def alfa_m(v):
  return (0.1 * (25 - v)) / tf.math.exp((25-v)/10) - 1

@tf.function
def beta_n(v):
  return 0.125 * tf.math.exp(-v/80)

@tf.function
def beta_h(v):
  return 1 / tf.math.exp((30-v)/10) + 1

@tf.function
def beta_m(v):
  return 4.0 * tf.math.exp(-v/18)

@tf.function
def I_na(v, m, h):
  return constant_dict.get('g_na') * tf.math.pow(m, 3) * h * (constant_dict.get('v_na') - v)

@tf.function
def I_k(v, n):
  return constant_dict.get('g_k') * tf.math.pow(n, 4) * (constant_dict.get('v_k') - v)

@tf.function
def I_L(v):
  return constant_dict.get('g_l') * (constant_dict.get('v_l') - v)

@tf.function
def HH_residual_calculator(t, u, u_t, model):
    """HH residual calculation.

    Args:
    ----
    t: temporal coordinate
    u: input function evaluated at discrete temporal coordinates
    u_t: input function evaluated at t
    model: DeepONet model

    Outputs:
    --------
    ODE_residual: residual of the governing ODE
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        preds = model({"forcing": u, "time": t})
        v_pred = preds[0]
        m_pred = preds[1]
        h_pred = preds[2]
        n_pred = preds[3]

    # Calculate gradients
    dv_dt = tape.gradient(v_pred, t)#[0:50]
    dm_dt = tape.gradient(m_pred, t)#[50:100]
    dh_dt = tape.gradient(h_pred, t)#[100:150]
    dn_dt = tape.gradient(n_pred, t)#[150:200]

    # ODE residual
    v_residual = constant_dict.get('Cm') * dv_dt - u_t - I_na(v_pred, m_pred, h_pred) - I_k(v_pred, n_pred) - I_L(v_pred) #u_t -
    m_residual = dm_dt - (alfa_m(v_pred) * (1 - m_pred)) - beta_m(v_pred) * m_pred
    h_residual = dh_dt - (alfa_h(v_pred) * (1 - h_pred)) - beta_h(v_pred) * h_pred
    n_residual = dn_dt - (alfa_n(v_pred) * (1 - n_pred)) - beta_n(v_pred) * n_pred

    return v_residual, m_residual, h_residual, n_residual