from mlProject.config.configuration import DataTransformationConfig
import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    # Transformation method: train test split 
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and testing sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splitted data into training and test sets")
        logger.info(f"Shape of training set: {train.shape}")
        logger.info(f"Shape of test sets: {test.shape}")
