import os
import yaml

class Config:


    def __init__(self, path: str = "config/config.yaml"):
        self.path = path
        self.cfg = None

    def load(self):
        """Load config.yaml file"""
        with open(self.path, 'r') as file:
            self.cfg = yaml.safe_load(file)
        return self.cfg

    def save(self, save_dir):
        """Save config file as yaml"""
        path = os.path.join(save_dir, "config.yaml")
        with open(path, 'w') as file:
            yaml.dump(self.cfg, file)