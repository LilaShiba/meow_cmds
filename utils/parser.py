import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score


class Parser:
    '''
    Parse sensor data and perform analysis tasks such as pair plots, linear regression, and KNN classification.
    '''
    def __init__(self, folder_path):
        '''
        Initialize the parser with the folder containing data files.
        
        Args:
            folder_path (str): Path to the folder containing CSV data files.
        '''
        self.folder_path = folder_path
        self.df = self.load_data()

    def load_data(self):
        '''
        Load all CSV files in the provided folder path into a single DataFrame.
        '''
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(self.folder_path, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in '{self.folder_path}'.")
            return pd.DataFrame()
        
        data_frames = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if data_frames:
            df_combined = pd.concat(data_frames, ignore_index=True)
            print(f"Loaded {len(data_frames)} files with {len(df_combined)} rows.")
            return df_combined
        else:
            print("No valid data was loaded.")
            return pd.DataFrame()

    def add_data(self, file_path):
        '''
        Append new data from a CSV file to the existing DataFrame.
        
        Args:
            file_path (str): Path to the CSV file to append.
        '''
        try:
            new_data = pd.read_csv(file_path)
            self.df = pd.concat([self.df, new_data], ignore_index=True) if not self.df.empty else new_data
            print(f"Appended data from {file_path}. Total rows: {len(self.df)}.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    def show_data(self, n=5):
        '''
        Display the first few rows of the DataFrame for inspection.
        
        Args:
            n (int): Number of rows to display.
        '''
        print(self.df.head(n) if not self.df.empty else "No data to display.")

    def plot_pair(self, columns=None):
        '''
        Generate a pair plot for selected columns of the data.
        
        Args:
            columns (list): List of column names to plot (default: all columns).
        '''
        if self.df.empty:
            print("No data to plot.")
            return
        
        columns = columns or self.df.columns
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return

        sns.pairplot(self.df[columns])
        plt.show()

    # def plot_heatmap(self):
    #     '''
    #     Generate a heatmap of correlations in the data.
    #     '''
    #     if self.df.empty:
    #         print("No data to plot.")
    #         return

    #     sns.heatmap(self.df.corr(), annot=False, cmap='coolwarm')
    #     plt.show()

    def plot_heatmap(self) -> None:
        """
        Plots a heatmap with trans flag colors.

        Args:
            df (pd.DataFrame): The data to visualize.
            title (str): The title of the heatmap.
        """
        # Trans flag colors: pastel blue, pastel pink, white
        trans_cmap = sns.color_palette(["#55CDFC", "#F7A8B8", "#FFFFFF", "#F7A8B8", "#55CDFC"])

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.iloc[:,1:].corr(), cmap=trans_cmap, annot=True, fmt=".2f", linewidths=0.5, cbar=True)

        plt.xlabel("Columns", fontsize=12)
        plt.ylabel("Rows", fontsize=12)

        plt.show()
        
    def linear_regression(self, target_column, feature_columns, verbose=True):
        '''
        Perform linear regression on selected features and target.
        
        Args:
            target_column (str): Name of the target variable.
            feature_columns (list): List of feature column names.
            verbose (bool): Print model details (default: True).
        
        Returns:
            model (LinearRegression): Fitted regression model.
        '''
        if self.df.empty:
            print("No data available for regression.")
            return None
        
        missing_cols = [col for col in [target_column] + feature_columns if col not in self.df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None

        X = self.df[feature_columns]
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        if verbose:
            print(f'Mean Squared Error: {mse:.4f}')
            print(f'Root Mean Squared Error: {rmse:.4f}')
            print(f'Regression Coefficients: {model.coef_}')
            print(f'Intercept: {model.intercept_}')
        
        return model

    def knn_classification(self, target_column, feature_columns, n_neighbors=3, verbose=True):
        '''
        Perform K-Nearest Neighbors classification on the data.
        
        Args:
            target_column (str): Name of the target variable.
            feature_columns (list): List of feature column names.
            n_neighbors (int): Number of neighbors (default: 3).
            verbose (bool): Print accuracy details (default: True).
        
        Returns:
            model (KNeighborsClassifier): Fitted KNN model.
        '''
        if self.df.empty:
            print("No data available for classification.")
            return None
        
        missing_cols = [col for col in [target_column] + feature_columns if col not in self.df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None
        
        X = self.df[feature_columns]
        y = self.df[target_column]

        if not pd.api.types.is_categorical_dtype(y) and not pd.api.types.is_integer_dtype(y):
            y = y.astype('category')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if verbose:
            print(f'Accuracy: {accuracy:.4f}')
        
        return model

    def knn_regression(self, target_column, feature_columns, n_neighbors=3, verbose=True):
        '''
        Perform K-Nearest Neighbors regression on the data.
        
        Args:
            target_column (str): Name of the target variable.
            feature_columns (list): List of feature column names.
            n_neighbors (int): Number of neighbors (default: 3).
            verbose (bool): Print performance details (default: True).
        
        Returns:
            model (KNeighborsRegressor): Fitted KNN model.
        '''
        if self.df.empty:
            print("No data available for regression.")
            return None
        
        missing_cols = [col for col in [target_column] + feature_columns if col not in self.df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None
        
        X = self.df[feature_columns]
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        if verbose:
            print(f'Mean Squared Error: {mse:.4f}')
            print(f'Root Mean Squared Error: {rmse:.4f}')
        
        return model


# Example usage
if __name__ == "__main__":
    data_folder = 'path_to_your_folder_with_data'
    parser = DataParser(data_folder)

    parser.show_data()  # Show first few rows of data
    parser.plot_pair()  # Plot pair plot
    parser.linear_regression('Temp_LSM9DS1', ['Accel_X', 'Accel_Y', 'Accel_Z'])
    parser.knn_classification('Proximity', ['Accel_X', 'Accel_Y', 'Accel_Z'])
