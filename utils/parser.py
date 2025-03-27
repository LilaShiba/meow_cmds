import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

class Parser:
    '''
    Parse sensor data and perform analysis tasks such as pair plots, linear regression, and KNN classification.
    '''
    def __init__(self, folder_path):
        '''
        Initialize the parser with the folder containing data files.
        '''
        self.folder_path = folder_path
        self.df = pd.DataFrame()
        self.load_data()

    def load_data(self):
        '''
        Load all CSV files in the provided folder path into a single DataFrame.
        '''
        data_frames = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.folder_path, file_name)
                df = pd.read_csv(file_path)
                data_frames.append(df)
        if data_frames:
            self.df = pd.concat(data_frames, ignore_index=True)

    def pair_plot(self, columns=None):
        '''
        Generate a pair plot for the selected columns of the data.
        '''
        if columns is None:
            columns = self.df.columns  # Default to all columns
        sns.pairplot(self.df[columns])
        plt.show()

    def linear_regression(self, target_column, feature_columns):
        '''
        Perform linear regression on the data using selected features and target.
        '''
        X = self.df[feature_columns]
        y = self.df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')

        # Display regression coefficients and intercept
        print(f'Regression Coefficients: {model.coef_}')
        print(f'Intercept: {model.intercept_}')

        return model

    def knn_classification(self, target_column, feature_columns, n_neighbors=3):
        '''
        Perform K-Nearest Neighbors classification on the data.
        '''
        X = self.df[feature_columns]
        y = self.df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')

        return model

# Example usage
if __name__ == "__main__":
    parser = Parser('path_to_your_folder_with_data')

    # Generate pair plot for all columns
    parser.pair_plot()

    # Perform linear regression on a target variable (e.g., 'Temp_LSM9DS1') with selected features
    parser.linear_regression(target_column='Temp_LSM9DS1', feature_columns=['Accel_X', 'Accel_Y', 'Accel_Z'])

    # Perform KNN classification on a target variable (e.g., 'Proximity') with selected features
    parser.knn_classification(target_column='Proximity', feature_columns=['Accel_X', 'Accel_Y', 'Accel_Z'])
