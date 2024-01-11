import types
import numpy as np

from scipy import signal
from brian2 import *


def calculate_HH_model_response(input_func, t_in_ms, mode='single'):
  start_scope()

  if isinstance(input_func, types.FunctionType):
    t_recorded = arange(int(t_in_ms*ms/defaultclock.dt))*defaultclock.dt/ms
    I_recorded = TimedArray(np.random.uniform(low=0, high=0.9)*500000*input_func(np.random.uniform(low=0.5, high=0.9)  * t_recorded), dt=defaultclock.dt)

  elif isinstance(input_func, np.ndarray):
    I_recorded = TimedArray(input_func, dt=defaultclock.dt)

  tresh = -40 * mV
  refrac = 3 * ms

  E_synap = -75 * mV
  conduct = 40 * nS

  Cm = 1 * uF  # Membrane capacitance
  g_na = 120 * msiemens  # Sodium conductance
  g_kd = 36 * msiemens  # Potassium conductance
  gl = 0.3 * msiemens  # Leak conductance
  ENa = 50 * mV  # Sodium reversal potential
  EK = -77 * mV  # Potassium reversal potential
  El = -54.4 * mV  # Leak reversal potential
  VRest = -65 * mV  # Resting membrane potential

  eqs_HH = '''
  dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm * second: volt
  dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/(exp((13.*mV-v+VT)/(4.*mV))-1.)*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/(exp((v-VT-40.*mV)/(5.*mV))-1.)*m : 1
  dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/(exp((15.*mV-v+VT)/(5.*mV))-1.)*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))*n : 1
  dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))*h : 1
  I = I_recorded(t) * nA: amp
  '''
  
  neuron_eqs = '''
  dv/dt = (1/Cm) * (I_Na + I_K + I_L + I) : volt
  I_Na = g_na * m**3 * h * (ENa - v) : amp
  I_K = g_kd * n**4 * (EK - v) : amp
  I_L = gl * (El - v) : amp
  dm/dt = (alpha_m * (1 - m) - beta_m * m) : 1
  dn/dt = (alpha_n * (1 - n) - beta_n * n) : 1
  dh/dt = (alpha_h * (1 - h) - beta_h * h) : 1
  alpha_m = (0.1/mV) * (10*mV) / exprel((v + 25*mV) / (10*mV))/ms : Hz
  beta_m = 4 * exp((v + 50*mV) / (18*mV))/ms : Hz
  alpha_n = (0.01/mV) * (10*mV) / exprel((v + 10*mV) / (10*mV))/ms : Hz
  beta_n = 0.125 * exp(v / (80*mV))/ms : Hz
  alpha_h = 0.07 * exp(v / (20*mV))/ms : Hz
  beta_h = 1 / (exp((v + 30*mV) / (10*mV)) + 1)/ms : Hz
  I = I_recorded(t) * nA: amp
  '''
  
  eqs_HH_post = '''
  dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
  dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
      (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
      (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
  dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
      (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
  dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
  I : amp
  '''

  group0 = NeuronGroup(1,
                       neuron_eqs,
                       threshold='v > tresh',
                       refractory=refrac,
                       method='exponential_euler')

  stm0 = StateMonitor(group0, variables=True, record=True)

  if mode == 'synapse':
    group1 = NeuronGroup(1, eqs_HH_post,
                        threshold='v > tresh',
                        refractory=refrac,
                        method='exponential_euler')

    S = Synapses(group0, group1, '''
                w : siemens (constant) # gap junction conductance
                I_post = w * (v_pre - E_synap) : amp (summed)
                ''')
    S.connect(i=[0], j=[0])
    S.w = conduct

    stm1 = StateMonitor(group1, variables=True, record=True)

  run(t_in_ms*ms)

  if mode == 'single': # /mV /nA
    return np.array(stm0.v[0]/mV), np.array(stm0.m[0]), np.array(stm0.n[0]), np.array(stm0.h[0]), np.array(stm0.I[0]/nA), np.array(stm0.t/ms)

  elif mode == 'synapse':
    return np.array(stm0.v[0]/mV), np.array(stm0.h[0]), np.array(stm0.I[0]/nA), np.array(stm1.v[0]/mV), np.array(stm1.h[0]), np.array(stm1.I[0]/nA), np.array(stm0.t/ms)

  elif mode == 'single-response':
    return np.array(stm0.v[0]/mV), np.array(stm0.m[0]), np.array(stm0.n[0]), np.array(stm0.h[0]), np.array(stm0.t/ms)


def generate_dataset(N_samples, t_in_ms, N_t_samples, inp_func, synaptic=False):
  generated_inputs = np.zeros((N_samples, t_in_ms * 10))
  generated_t_inputs = np.zeros((N_samples, t_in_ms * 10))

  generated_v_responses = np.zeros((N_samples, t_in_ms * 10))
  generated_m_responses = np.zeros((N_samples, t_in_ms * 10))
  generated_n_responses = np.zeros((N_samples, t_in_ms * 10))
  generated_h_responses = np.zeros((N_samples, t_in_ms * 10))
  generated_i_responses = np.zeros((N_samples, t_in_ms * 10))
  generated_t_responses = np.zeros((N_samples, t_in_ms * 10))

  for i in range(N_samples//2):
    _, _, _, _, _, generated_inputs[i, :], generated_t_inputs[i, :] = calculate_HH_model_response(inp_func, t_in_ms, mode='synapse')
    generated_v_responses[i, :], generated_m_responses[i, :], generated_n_responses[i, :], generated_h_responses[i, :], generated_i_responses[i, :], generated_t_responses[i, :] = calculate_HH_model_response(generated_inputs[i, :], t_in_ms)

  if synaptic == True:
    for u in range(N_samples//2, N_samples):
      generated_inputs[u, :] = np.random.uniform(low=0, high=0.9) * inp_func(np.random.uniform(low=0.05, high=0.2) * (np.pi / 2) * generated_t_responses[0, :])
      generated_t_inputs[u, :] = generated_t_responses[0, :]
      generated_v_responses[u, :], generated_m_responses[u, :], generated_n_responses[u, :], generated_h_responses[u, :], generated_i_responses[u, :], generated_t_responses[u, :] = calculate_HH_model_response(generated_inputs[u, :], t_in_ms)
  else:
    for u in range(N_samples//2):
      _, _, _, _, _, generated_inputs[u, :], generated_t_inputs[u, :] = calculate_HH_model_response(inp_func, t_in_ms, mode='synapse')
      generated_v_responses[u, :], generated_m_responses[u, :], generated_n_responses[u, :], generated_h_responses[u, :], generated_i_responses[u, :], generated_t_responses[u, :] = calculate_HH_model_response(generated_inputs[u, :], t_in_ms)

  X = np.zeros((N_samples*N_t_samples, t_in_ms * 10 + 2))
  y = np.zeros((N_samples*N_t_samples, 4))
  t = generated_t_responses[0, :]

  id = -1
  for o in range(N_samples):
    for p in range(N_t_samples):
      if N_t_samples != t_in_ms*10:
        id = np.random.choice(range(len(t)))
      else:
        id += 1
      X[p+o*N_t_samples, : 1] = t[id]
      X[p+o*N_t_samples, 1 : t_in_ms * 10+1] = generated_inputs[o, :]
      X[p+o*N_t_samples, t_in_ms * 10+1 : t_in_ms * 10+2] = generated_inputs[o, id]
      y[p+o*N_t_samples, :] = generated_v_responses[o, id], generated_m_responses[o, id], generated_n_responses[o, id], generated_h_responses[o, id]

  shuffler = np.random.permutation(N_samples*N_t_samples)

  return X[shuffler, :], y[shuffler, :], t