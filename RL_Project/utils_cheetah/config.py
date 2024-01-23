import os
import yaml

class Config:

    def __init__(self, dir: str= "config", filename: str= "dummy_config.yaml"):
        self.path = os.path.join(dir, filename)
        self.cfg = None

    def load(self):
        with open(self.path, 'r') as file:
            self.cfg = yaml.safe_load(file)
        return self.cfg

    def save(self, save_dir):
        path = os.path.join(save_dir, "config.yaml")
        with open(path, 'w') as file:
            yaml.dump(self.cfg, file)