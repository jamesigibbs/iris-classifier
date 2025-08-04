# tests/test_train.py
import os
from src.train import train_model

def test_model_saves():
    path = "outputs/test_model.pkl"
    if os.path.exists(path):
        os.remove(path)
    train_model(save_path=path)
    assert os.path.exists(path)