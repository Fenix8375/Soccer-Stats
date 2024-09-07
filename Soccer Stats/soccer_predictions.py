import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load all CSV files into a list of DataFrames
folder_path = "./Prem Data/Datasets"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
dfs = [pd.read_csv(file, index_col=0) for file in csv_files]

combined_df = pd.concat(dfs, ignore_index=True)

# Convert the Date column to datetime format and handle invalid dates
combined_df["Date"] = pd.to_datetime(combined_df["Date"], format="%d/%m/%Y", dayfirst=True, errors='coerce')
combined_df = combined_df.dropna(subset=["Date"])

# Create 'season' column based on the year of the 'Date' column
combined_df["season"] = combined_df["Date"].dt.year

# Reset index after concatenation to avoid fragmentation warning
new_combined_df_reset = combined_df.reset_index(drop=False)

# Function to add a new column based on the 'FTR = Full Time Result' column
def adding_team(team):
    team["target"] = team["FTR"].shift(-1)
    return team

# Group by 'HomeTeam', 'AwayTeam', and 'season', and apply the 'adding_team' function
new_combined_df = new_combined_df_reset.groupby(["HomeTeam", "AwayTeam", "season"], group_keys=False).apply(adding_team)

# Fill NaN values in 'target' column with a default value
new_combined_df["target"] = new_combined_df["target"].fillna("D")

# Map 'H', 'A', 'D' in the 'target' column to numeric values
target_mapping = {'H': 0, 'A': 1, 'D': 2}
new_combined_df["target"] = new_combined_df["target"].replace(target_mapping).astype(int)

# Drop rows with missing target values
new_combined_df = new_combined_df.dropna(subset=["target"])

# Initialize RidgeClassifier with class weights to handle imbalance
rr = RidgeClassifier(alpha=1, class_weight='balanced')
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split, n_jobs=1)

# Remove irrelevant columns and keep only numeric ones for imputation and scaling
remove_columns = ["HomeTeam", "target", "AwayTeam", "HTR", "Attendance", "Referee", "Time", "FTR", "season", "Date", "_date"]
selected_columns = new_combined_df.columns[~new_combined_df.columns.isin(remove_columns)]

# Select only numeric columns for imputation
numeric_columns = new_combined_df[selected_columns].select_dtypes(include=[np.number]).columns.tolist()

# Remove columns that have all missing values
non_empty_numeric_columns = [col for col in numeric_columns if new_combined_df[col].notna().sum() > 0]

# Handle columns with NaNs and impute missing values only in non-empty numeric columns
imputer = SimpleImputer(strategy="mean")
new_combined_df[non_empty_numeric_columns] = imputer.fit_transform(new_combined_df[non_empty_numeric_columns])

# Scale the non-empty numeric columns
scaler = MinMaxScaler()
new_combined_df[non_empty_numeric_columns] = scaler.fit_transform(new_combined_df[non_empty_numeric_columns])

# Perform feature selection using only non-empty numeric columns
sfs.fit(new_combined_df[non_empty_numeric_columns], new_combined_df["target"])

# Fix: Ensure proper alignment of selected columns
predictors = [col for col, selected in zip(non_empty_numeric_columns, sfs.get_support()) if selected]

# Backend function to train and predict
def backend(data, model, predictions, start=2, step=1):
    all_predictions = []
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        
        if train.empty or test.empty:
            continue
        
        model.fit(train[predictions], train["target"])
        preds = model.predict(test[predictions])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        all_predictions.append(combined)
    
    if not all_predictions:
        raise ValueError("No predictions were made, check your data splits.")
    
    return pd.concat(all_predictions)

# Run the backend to get predictions
predictions = backend(new_combined_df, rr, predictors)

# Evaluate the performance using accuracy
best_score = accuracy_score(predictions["actual"], predictions["prediction"])
print("Accuracy Score:", best_score)

# Count home wins and away wins
home_wins = new_combined_df[new_combined_df["FTR"] == "H"].groupby("HomeTeam").size()
away_wins = new_combined_df[new_combined_df["FTR"] == "A"].groupby("AwayTeam").size()

# Merge the home and away win counts and fill missing values with 0
team_wins = pd.concat([home_wins, away_wins], axis=1, keys=["home_wins", "away_wins"]).fillna(0)

# Calculate the total wins for each team
team_wins["total_wins"] = team_wins["home_wins"] + team_wins["away_wins"]

# Apply the rolling mean calculation to the numeric columns
def finding_average(team):
    return team.rolling(10).mean()

combined_df_rolling = finding_average(team_wins)

# Update column names for rolling averages
rolling_columns = [f"{col}_10" for col in combined_df_rolling.columns]
combined_df_rolling.columns = rolling_columns

# Concatenate rolling averages with the main DataFrame
new_combined_df = pd.concat([new_combined_df, combined_df_rolling], axis=1)

# Shift columns for next team predictions
def adding_columns(df, col):
    return df.groupby("season", group_keys=False)[col].apply(lambda x: x.shift(-1))

new_combined_df["home_next"] = adding_columns(new_combined_df, "HomeTeam")
new_combined_df["team_opp_next"] = adding_columns(new_combined_df, "AwayTeam")
new_combined_df["next_date"] = adding_columns(new_combined_df, "Date")

new_combined_df = new_combined_df.dropna(subset=["next_date"])

# Remove unnecessary columns
new_combined_df = new_combined_df.drop(columns=["index", "Attendance", "Referee", "Date"])

# Merge dataframes and handle missing values
full = new_combined_df.merge(
    new_combined_df[rolling_columns + ["team_opp_next", "next_date", "HomeTeam"]],
    left_on=["HomeTeam", "next_date"],
    right_on=["team_opp_next", "next_date"],
    how="inner"
)

# Clean full dataframe by dropping columns with all NaN values
full_clean = full.dropna(axis=1, how='all')

# Select only numeric columns for imputation
numeric_columns_full = full_clean.select_dtypes(include=[np.number]).columns.tolist()

# Impute missing values and perform feature selection
X = full_clean[numeric_columns_full]
X_imputed = imputer.fit_transform(X)

valid_columns = [col for col, keep in zip(numeric_columns_full, imputer.statistics_) if not np.isnan(keep)]
X_imputed_df = pd.DataFrame(X_imputed, columns=valid_columns)

# Perform feature selection
sfs.fit(X_imputed_df, full_clean["target"])

# Print selected features
selected_features = X_imputed_df.columns[sfs.get_support()]
# print("Selected features:", selected_features)

predictions = backend(full,rr,predictors)

result = accuracy_score(predictions["actual"], predictions["prediction"])

print(result)
