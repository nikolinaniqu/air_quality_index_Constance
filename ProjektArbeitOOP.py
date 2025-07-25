# ==============================================================================
# Project Overview:
# This script loads air quality data from a CSV file, processes it, calculates
# air quality indices (AQI) for various pollutants, and visualizes the data.
# It also includes a machine learning model (Random Forest Regressor) to predict
# the overall AQI and evaluates its performance.
# ==============================================================================
# Required Libraries:
# - csv,numpy and pandas for data manipulation
# - seaborn and matplotlib for plotting
# - sklearn for machine learning and model evaluation
# ==============================================================================
# Author: Nikolina Nisic-Quinones
# Date: 24.07.2025
# ==============================================================================
# Import necessary libraries
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Loads and cleans the air quality data from a semicolon-delimited CSV file.
    """
    def __init__(self, file):
        """
         Initializes the loader with the path of the dataset
        :param file: path to the CSV file
        """
        self.file = file


    def load_and_clean_data(self):
        """
        Loads the raw CSV data, separates columns and rows into two different lists,
         cleans invalid entries, parses dates, and converts pollutant values to numeric.
        :return: Cleaned DataFrame with pollutants and date columns in appropriate formats.
        """
        rows = []
        with open(self.file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=';', quotechar='"')
            for row in reader:
                if not row or "Stationscode" in row[0]:
                    continue
                rows.append(row)

        cleaned_rows = []
        for row in rows:
            row = row[0].strip().replace('"', '').replace("'", "").replace(",", "").replace("-", str(np.nan))
            fields = row.split(";")
            if len(fields) == 7:
                cleaned_rows.append(fields)

        columns = ["Stationscode", "Datum", "PM10", "O3", "NO2", "PM2_5", "Luftqualitätsindex"]
        self.df = pd.DataFrame(cleaned_rows, columns=columns)

        # Convert the date column to datetime
        self.df["Datum"] = pd.to_datetime(self.df["Datum"].str.split(" ").str[0], format="%d.%m.%Y")

        # Convert pollutants values to numeric
        self.df["PM10"] = pd.to_numeric(self.df["PM10"], errors="coerce")
        self.df["O3"] = pd.to_numeric(self.df["O3"], errors="coerce")
        self.df["PM2_5"] = pd.to_numeric(self.df["PM2_5"], errors="coerce")
        self.df["NO2"] = pd.to_numeric(self.df["NO2"], errors="coerce")
        self.df.dropna(inplace=True)

        return self.df


class AirQualityIndex:
    """
    Calculates the air quality index for the following pollutants: PM10, PM2.5,O3 and NO2
    as well as the overall air quality index.
    """
    def __init__(self, df):
        self.df = df
        self.aqi_no2 = {
            "sehr schlecht": (201, np.inf),
            "schlecht": (101, 200),
            "mäßig": (41, 100),
            "gut": (21, 40),
            "sehr gut": (0, 20),
        }
        self.aqi_pm10 = {
            "sehr schlecht": (101, np.inf),
            "schlecht": (51, 100),
            "mäßig": (36, 50),
            "gut": (21, 35),
            "sehr gut": (0, 20)
        }
        self.aqi_pm2_5 = {
            "sehr schlecht": (51, np.inf),
            "schlecht": (26, 50),
            "mäßig": (21, 25),
            "gut": (11, 20),
            "sehr gut": (0, 10)
        }
        self.aqi_o3 = {
            "sehr schlecht": (241, np.inf),
            "schlecht": (181, 240),
            "mäßig": (121, 180),
            "gut": (61, 120),
            "sehr gut": (0, 60),
        }

    def calculate_aqi(self, value, aqi_dict):
        """
        Calculates the air quality index (AQI) for a given pollutant based on its concentration value
        and the provided AQI category ranges.

        :param value: (float) Pollutant concentration value.
        :param aqi_dict: (dict) Dictionary with AQI category ranges for the pollutant.
        :return: (float) Calculated AQI value, or None if the value is invalid
        """

        for category, (low, high) in aqi_dict.items():
            if low < value <= high:
                I_low, I_high = self.get_aqi_range(category)
                return (I_high - I_low) / (high - low) * (value - low) + I_low

    def get_aqi_range(self, category):
        """
         Provides index aqi range based on the AQI category.
        :param category: (str)AQI category.
        :return: tuple(min, max) AQI value for the category.
        """
        if category == "sehr gut":
            return 0, 50
        elif category == "gut":
            return 51, 100
        elif category == "mäßig":
            return 101, 150
        elif category == "schlecht":
            return 151, 200
        elif category == "sehr schlecht":
            return 201, 500
        return 0, 50

    def calculate_overall_aqi(self):
        """
        Calculates individual AQI values for each pollutant (PM10, PM2.5, O3, NO2)
        and computes the overall AQI for each row in the DataFrame.
        The overall AQI is set to the highest AQI value among the pollutants.
        :return: pd.DataFrame: DataFrame with columns for individual AQIs and overall AQI.
        """
        def calc(row):
            pollutants = {
                "PM10": row["PM10"],
                "PM2_5": row["PM2_5"],
                "O3": row["O3"],
                "NO2": row["NO2"]
            }
            aqi_values = []
            for pollutant, value in pollutants.items():
                if pollutant == "PM10":
                    aqi_values.append(self.calculate_aqi(value, self.aqi_pm10))
                elif pollutant == "PM2_5":
                    aqi_values.append(self.calculate_aqi(value, self.aqi_pm2_5))
                elif pollutant == "O3":
                    aqi_values.append(self.calculate_aqi(value, self.aqi_o3))
                elif pollutant == "NO2":
                    aqi_values.append(self.calculate_aqi(value, self.aqi_no2))

            valid_aqi_values = [value for value in aqi_values if value is not None]
            if valid_aqi_values:
                overall_aqi = max(valid_aqi_values)
                return pd.Series({
                    "AQI_PM10": aqi_values[0],
                    "AQI_PM2_5": aqi_values[1],
                    "AQI_O3": aqi_values[2],
                    "AQI_NO2": aqi_values[3],
                    "OverallAQI": overall_aqi
                })
            #Return None for the uncalculated rows
            return pd.Series([None] * 5, index=["AQI_PM10", "AQI_PM2_5", "AQI_O3", "AQI_NO2", "OverallAQI"])

        self.df[["AQI_PM10", "AQI_PM2_5", "AQI_O3", "AQI_NO2", "OverallAQI"]] = self.df.apply(calc, axis=1)

        return self.df
    def export_pdSeries(self,df):
        """
        Exports the Data Frame in JSON Format.
        :return: (str) JSON string path (File is saved on the disk)
        """
        json=df.to_json("AirQualityIDx_Constance.json",force_ascii=False,orient="index")
class Visualizer():
    """
    Visualizes and analyzes the dataset using the heatmap, scatterplot and lineplot.
    """
    def __init__(self, df,y_test,y_pred):
        self.df = df
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_data(self,df,y_test,y_pred,corr):
        """
        Visualizes AQI data through multiple plots:

        1. Time-series scatter plots of AQI for each pollutant (PM10, PM2.5, NO2, O3)
       and overall AQI over time.
        2. A correlation heatmap to display the relationship between AQI values for
       different pollutants.
        3. A line plot comparing actual vs predicted AQI values.

        Adjustments to plot appearance:
        - Scatter and line plots for AQI time series.
        - Heatmap with correlation values annotated.
        - Custom axis labels, font sizes, and legend positioning.

        :return: None
        """
        sample_df= self.df.sample(frac=0.3,random_state=42)

        fig=plt.figure(figsize=(16,8))
        gs=GridSpec(nrows=2,ncols=2, figure=fig,width_ratios=[2,1],height_ratios=[0.8,1])
        #AQI TimeSeries plot
        ax1=fig.add_subplot(gs[:,0])
        ax1.set_title("Jahresübersicht des Luftqualitätsindex (AQI) für die Stadt Konstanz\n im Zeitraum vom 1. Januar 2024 bis 21. Juli 2025")
        sns.scatterplot(data=sample_df,x=df["Datum"] ,y=df["AQI_PM10"].sample(frac=0.15,random_state=42),ax=ax1, label="PM10",s=13)
        sns.scatterplot(data=sample_df, x=df["Datum"], y=df["AQI_PM2_5"].sample(frac=0.15,random_state=42), ax=ax1, label="PM2.5",s=13)
        sns.scatterplot(data=sample_df, x=df["Datum"], y=df["AQI_NO2"].sample(frac=0.1,random_state=42), ax=ax1, label="NO2",s=13)
        sns.scatterplot(data=sample_df, x=df["Datum"], y=df["AQI_O3"].sample(frac=0.1,random_state=42), ax=ax1, label="O3",s=13)
        sns.lineplot(data=sample_df, x=df["Datum"], y=df["OverallAQI"],ax=ax1, label="Overall AQI")
        ax1.set_ylabel("Luftqualitätsindex (AQI)")
        ax1.set_xlabel("Datum")
        ax1.legend()
        # Correlation Heatmap of the AQI values
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("AQI-Korrelations-Heatmap")
        sns.heatmap(corr, cmap='coolwarm',annot=True, ax=ax2)
        plt.xticks(fontsize=6.5, rotation=40)
        plt.yticks(fontsize=6.5)
        #Plot the Actual vs predicted AQI model
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_title("Tatsächliche vs. Vorhergesagte AQI-Werte")
        ax3.plot(y_test.values[::10], label="Tatsächlicher AQI")
        ax3.plot(y_pred[::10], label="Vorhergesagter AQI", alpha=0.5)
        ax3.set_ylabel("Lüftqualitätsindex")
        ax3.set_xlabel("Proben")
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.show()


    def analyze_data(self,df):
        """
        Calculates the AQI values correlations.
        :return: Correlation matrix.
        """
        corr = df[["OverallAQI", "AQI_PM10", "AQI_PM2_5", "AQI_O3", "AQI_NO2"]].corr()
        return corr

class Modeler:
    """
    A class to train a Random Forest regression model to predict Overall AQI
    from individual pollutant AQI values.
    """
    def __init__(self, df):
        self.df = df

    def train_model(self):
        """Trains a Random Forest Regressor on AQI features and evaluates the model.

        The model uses 'AQI_PM10', 'AQI_PM2_5', 'AQI_O3', and 'AQI_NO2' as input
        features to predict 'OverallAQI'. After training, it prints performance metrics.
        :return: Predicted and actual values of the AQI
        """
        X = self.df[["AQI_PM10", "AQI_PM2_5", "AQI_O3", "AQI_NO2"]]
        y = self.df["OverallAQI"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))

        return y_test,y_pred

def main():
    # Load and clean data
    data_loader = DataLoader(file="Luftqualitaet_Konstanz01012024_21072025.csv")
    df = data_loader.load_and_clean_data()

    # Calculate AQI
    air_quality_index = AirQualityIndex(df)
    aqi_calculated = air_quality_index.calculate_overall_aqi()
    exported_file = air_quality_index.export_pdSeries(df)

    # Train model and get predictions
    data_model = Modeler(df)
    y_test, y_pred = data_model.train_model()

    # Visualize the data
    visualise_data = Visualizer(df, y_test, y_pred)
    analyzed_data = visualise_data.analyze_data(df)
    visualise_data.plot_data(df, y_test, y_pred, analyzed_data)

if __name__ == "__main__":
    main()