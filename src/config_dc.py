import os
import typing as t
import tensorflow as tf

from datetime import datetime
from dataclasses import asdict, dataclass, field, fields
from dataclasses_json import dataclass_json
from scipy import signal


@dataclass_json
@dataclass
class Config:
    input_function: t.Callable = signal.square
    N_samples_train: int = 500
    N_samples_val: int = 100
    N_samples_test: int = 100
    N_t_samples: int = 100
    t_in_ms: int = 100
    n_epochs: int = 500
    col_batch_size: int = 2000
    learning_rate: float = 1e-6
    ini_batch_size:  int = 20
    IC_weight: tf.constant = tf.constant(2.0, dtype=tf.float32)
    v_weight: tf.constant = tf.constant(1.0, dtype=tf.float32)
    m_weight: tf.constant = tf.constant(1.0, dtype=tf.float32)
    h_weight: tf.constant = tf.constant(1.0, dtype=tf.float32)
    n_weight: tf.constant = tf.constant(1.0, dtype=tf.float32)
    ODE_weight= tf.constant = tf.constant(1.0, dtype=tf.float32)
    npy_res_path: str = str(os.path.join('results', 'EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.npy')) #zmienic na destination path czy cos
    cfg_path: str = str(os.path.join('data', 'configs', 'CFG_EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.json'))
    npy_dat_path: str = str(os.path.join('data', '....npy'))
    plot_response_path: str = str(os.path.join('results', 'EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'_response.png'))
    plot_history_path: str = str(os.path.join('results', 'EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'_history.png'))
    #df_columns: list = field(default_factory=list)
