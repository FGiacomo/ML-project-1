import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from nearZeroVariance import DataAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

class ClassificationDataPreparer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.df_clean = None
        self.df_encoded = None

        self.nominal_variables = [
            'person_gender', 'entity_type', 'channel',
            'agent_id', 'entity_a', 'location', 'product_id'
        ]

    def clean_data(self):
        print("Dropping rows with missing values (keeping all columns)...")
        self.df_clean = self.df.dropna().copy()

    def encode_nominal(self):
        print("One-hot encoding nominal variables...")
        self.df_encoded = pd.get_dummies(
            self.df_clean,
            columns=self.nominal_variables,
            drop_first=False,  # Keep all dummy columns to preserve full information
            dtype=int
        )

    def save_data(self, save_path_encoded, save_path_clean):
        print("Saving encoded and cleaned datasets...")
        with open(save_path_encoded, "wb") as f:
            pickle.dump(self.df_encoded, f)
        with open(save_path_clean, "wb") as f:
            pickle.dump(self.df_clean, f)

    def run_all(self, save_path_encoded, save_path_clean):
        self.clean_data()
        self.encode_nominal()
        self.save_data(save_path_encoded, save_path_clean)


class ClassificationTestPreparer:
    def __init__(self, filepath, nominal_variables, train_columns):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.df_clean = None
        self.df_encoded = None
        self.nominal_variables = nominal_variables
        self.train_columns = train_columns  # To align columns

    def clean_data(self):
        print("Dropping missing values in test data...")
        self.df_clean = self.df.dropna().copy()

    def encode_nominal(self):
        print("One-hot encoding nominal variables in test data...")
        df_encoded = pd.get_dummies(
            self.df_clean,
            columns=self.nominal_variables,
            drop_first=False,
            dtype=int
        )
        print("Aligning columns to match training data...")
        df_encoded = df_encoded.reindex(columns=self.train_columns, fill_value=0)
        self.df_encoded = df_encoded

    def save_data(self, save_path_encoded, save_path_clean):
        print("Saving test data...")
        with open(save_path_encoded, "wb") as f:
            pickle.dump(self.df_encoded, f)
        with open(save_path_clean, "wb") as f:
            pickle.dump(self.df_clean, f)

    def run_all(self, save_path_encoded, save_path_clean):
        self.clean_data()
        self.encode_nominal()
        self.save_data(save_path_encoded, save_path_clean)
