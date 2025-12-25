import os
import sys
from src.exception import  CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    artifacts_dir: str = "artifacts"
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _create_directories(self) -> None:
        """Ensure required directories exist."""
        os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Executes the data ingestion process:
        - Reads raw data
        - Saves raw data
        - Performs train-test split
        - Persists split datasets
        """
        logging.info("Starting data ingestion process")

        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Raw dataset loaded successfully")

            self._create_directories()

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to artifacts")

            train_df, test_df = train_test_split(
                df, test_size=0.25, random_state=42
            )
            logging.info("Train-test split completed")

            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and test datasets saved successfully.")
            logging.info("Ingestion of data completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as exc:
            logging.exception("Data ingestion failed")
            raise CustomException(exc,sys)


if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()