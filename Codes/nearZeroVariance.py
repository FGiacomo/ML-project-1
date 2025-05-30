# lets define a function checking near-zero variance
import pandas as pd

class DataAnalyzer:
    @staticmethod
    def near_zero_var(df, freq_cut=95/5, unique_cut=10):
        """
        Identifies columns with near-zero variance in a DataFrame and calculates indicators.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - freq_cut (float): Threshold for the frequency ratio (default = 95/5).
        - unique_cut (int): Threshold for the unique value ratio (default = 10).

        Returns:
        - pd.DataFrame: A sorted DataFrame containing:
            - variable: Column name
            - freq_ratio: Ratio of the most common value to the second most common value
            - unique_ratio: Ratio of unique values to total observations
            - high_freq_ratio: Binary indicator (1 if freq_ratio > freq_cut)
            - low_unique_ratio: Binary indicator (1 if unique_ratio < unique_cut)
        """
        results = []

        for col in df.columns:
            # Get the value counts
            counts = df[col].value_counts()

            # Calculate freq_ratio
            if len(counts) > 1:
                freq_ratio = counts.iloc[0] / counts.iloc[1]
            else:
                freq_ratio = float('inf')  # Only one unique value

            # Calculate unique_ratio
            unique_ratio = len(counts) / len(df)

            # Determine binary indicators
            high_freq_ratio = int(freq_ratio > freq_cut)
            low_unique_ratio = int(unique_ratio < unique_cut)

            # Append results
            results.append({
                'variable': col,
                'freq_ratio': freq_ratio,
                'unique_ratio': unique_ratio,
                'high_freq_ratio': high_freq_ratio,
                'low_unique_ratio': low_unique_ratio
            })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Sort by 'high_freq_ratio' (descending) and 'low_unique_ratio' (ascending)
        results_df = results_df.sort_values(by=['freq_ratio', 'unique_ratio'], ascending=[False, True])

        return results_df
