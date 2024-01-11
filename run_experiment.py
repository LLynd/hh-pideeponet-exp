from src.config_dc import Config
from src.generating import generate_dataset
from src.train import train
from src.visualizing import plot_history, plot_model_response


cfg = Config()

X_train, y_train, t_train = generate_dataset(cfg.N_samples_train, cfg.t_in_ms, cfg.N_t_samples, cfg.input_function)
X_val, y_val, t_val = generate_dataset(cfg.N_samples_val, cfg.t_in_ms, cfg.N_t_samples, cfg.input_function)
X_test, y_test, t_test = generate_dataset(cfg.N_samples_test, cfg.t_in_ms, cfg.N_t_samples, cfg.input_function)

train(X_train, y_train, X_val, y_val, cfg, X_test, y_test)
