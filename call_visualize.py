import pickle
from visualize import visualize
from train import model

# Load trained parameters
with open("params.pkl", "rb") as f:
    params = pickle.load(f)

visualize(params, model)