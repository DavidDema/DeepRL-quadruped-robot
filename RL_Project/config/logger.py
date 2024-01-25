import os
import pickle
import yaml

def load_cfg(path: str = "config/config.yaml"):
    """Load config.yaml file"""
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def save_cfg(cfg: str, save_dir: str):
    """Save config file as yaml"""
    path = os.path.join(save_dir, "config.yaml")
    with open(path, 'w') as file:
        yaml.dump(cfg, file)


def load_data(path: str="checkpoints/latest/data.pkl"):
    """Load data.pkl file"""
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_experiment(path: str = "checkpoints/latest"):
    data = load_data(os.path.join(path, "data.pkl"))
    cfg = load_cfg(os.path.join(path, "config.yaml"))
    return data, cfg
