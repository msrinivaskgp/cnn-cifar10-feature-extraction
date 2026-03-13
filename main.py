from src.train import train_model
from src.feature_extraction import extract_features

if __name__ == "__main__":

    model = train_model()

    extract_features(model)