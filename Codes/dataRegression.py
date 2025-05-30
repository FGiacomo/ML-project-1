# Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from nearZeroVariance import DataAnalyzer  # Assuming you have this module
import seaborn as sns
import matplotlib.pyplot as plt

class RegressionDataPreparer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.df_clean = None
        self.df_encoded = None
        self.nominal_variables = ['obj_type', 'own_type', 'build_mat', 'loc_code']
        self.ordinal_variables_to_transform = ['cond_class', 'build_mat', 'has_park', 'has_balcony',
                                               'has_lift', 'has_sec', 'has_store']
        self.ordinal_orders = {
            'cond_class': ['low', 'medium', 'good', 'top'],
            'build_mat': ['weak', 'medium', 'strong', 'very_strong'],
            'has_park': ['no', 'yes'],
            'has_balcony': ['no', 'yes'],
            'has_lift': ['no', 'yes'],
            'has_sec': ['no', 'yes'],
            'has_store': ['no', 'yes']
        }

    def clean_data(self):
        print("Dropping missing values only, retaining all columns...")
        self.df_clean = self.df.dropna().copy()

    def reduce_rare_categories(self):
        print("Reducing rare categories in nominal variables...")
        for var in self.nominal_variables:
            if var in self.df_clean.columns:
                value_counts = self.df_clean[var].value_counts()
                rare_levels = value_counts[value_counts <= 25].index
                self.df_clean[var] = self.df_clean[var].replace(rare_levels, "Other")

    def drop_nzv(self):
        print("Dropping near-zero variance variables...")
        nzv_summary = DataAnalyzer.near_zero_var(self.df_clean)
        variables_nzv = nzv_summary[
            (nzv_summary['low_unique_ratio'] == 1) & 
            (nzv_summary['high_freq_ratio'] == 1)
        ]['variable']
        self.df_clean.drop(variables_nzv, axis=1, inplace=True)
        self.nominal_variables = [var for var in self.nominal_variables if var not in variables_nzv.tolist()]

    def encode_ordinal(self):
        print("Encoding ordinal variables...")
        encoder = OrdinalEncoder(
            categories=[self.ordinal_orders[var] for var in self.ordinal_variables_to_transform if var in self.df_clean.columns],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        vars_to_encode = [var for var in self.ordinal_variables_to_transform if var in self.df_clean.columns]
        self.df_encoded = self.df_clean.copy()
        self.df_encoded[vars_to_encode] = encoder.fit_transform(self.df_encoded[vars_to_encode])

    def encode_nominal(self):
        print("Encoding nominal variables using one-hot encoding...")
        vars_to_encode = [var for var in self.nominal_variables if var in self.df_encoded.columns]
        self.df_encoded = pd.get_dummies(
            self.df_encoded,
            columns=vars_to_encode,
            drop_first=True,
            dtype=int
        )

    def save_data(self, save_path_encoded, save_path_clean):
        print("Saving encoded and cleaned datasets...")
        with open(save_path_encoded, "wb") as f:
            pickle.dump(self.df_encoded, f)
        with open(save_path_clean, "wb") as f:
            pickle.dump(self.df_clean, f)

    def plot_price_distribution(self):
        print("Plotting price_z distributions...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df_clean["price_z"], kde=True)
        plt.title("Original Distribution of price_z")

        Sale_Price_Log = np.log1p(self.df_clean["price_z"])
        plt.subplot(1, 2, 2)
        sns.histplot(Sale_Price_Log, kde=True)
        plt.title("Log-Transformed Distribution of price_z")

        plt.tight_layout()
        plt.show()

    def run_all(self, save_path_encoded, save_path_clean):
        self.clean_data()
        self.reduce_rare_categories()
        self.drop_nzv()
        self.encode_ordinal()
        self.encode_nominal()
        
        self.save_data(save_path_encoded, save_path_clean)

class TestDataPreparer:
    def __init__(self, filepath, ordinal_orders, nominal_variables):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.df_clean = None
        self.df_encoded = None
        self.ordinal_orders = ordinal_orders
        self.nominal_variables = nominal_variables
        self.ordinal_variables_to_transform = list(ordinal_orders.keys())

    def clean_data(self):
        print("Cleaning test data...")
        self.df_clean = self.df.dropna().copy()
        #self.df_clean.drop(columns=['unit_id', 'src_month', 'loc_code'], errors='ignore', inplace=True)

    def encode_ordinal(self):
        print("Encoding ordinal variables in test data...")
        encoder = OrdinalEncoder(
            categories=[self.ordinal_orders[var] for var in self.ordinal_variables_to_transform if var in self.df_clean.columns],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        vars_to_encode = [var for var in self.ordinal_variables_to_transform if var in self.df_clean.columns]
        self.df_encoded = self.df_clean.copy()
        self.df_encoded[vars_to_encode] = encoder.fit_transform(self.df_encoded[vars_to_encode])

    def encode_nominal(self):
        print("One-hot encoding nominal variables in test data...")
        vars_to_encode = [var for var in self.nominal_variables if var in self.df_encoded.columns]
        self.df_encoded = pd.get_dummies(
            self.df_encoded,
            columns=vars_to_encode,
            drop_first=True,
            dtype=int
        )

    def save_data(self, save_path_encoded, save_path_clean):
        print("Saving test data...")
        with open(save_path_encoded, "wb") as f:
            pickle.dump(self.df_encoded, f)
        with open(save_path_clean, "wb") as f:
            pickle.dump(self.df_clean, f)

    def run_all(self, save_path_encoded, save_path_clean):
        self.clean_data()
        self.encode_ordinal()
        self.encode_nominal()
        self.save_data(save_path_encoded, save_path_clean)

# Example usage for training:
# preparer = RegressionDataPreparer("../data/appartments_train.csv")
# preparer.run_all(
#     save_path_encoded="../data/prepared_files/train/regression_train_encoded.pkl",
#     save_path_clean="../data/prepared_files/train/regression_train.pkl"
# )

# Example usage for test:
# test_preparer = TestDataPreparer(
#     filepath="../data/appartments_test.csv",
#     ordinal_orders=preparer.ordinal_orders,
#     nominal_variables=preparer.nominal_variables
# )
# test_preparer.run_all(
#     save_path_encoded="../data/prepared_files/test/regression_test_encoded.pkl",
#     save_path_clean="../data/prepared_files/test/regression_test.pkl"
# )