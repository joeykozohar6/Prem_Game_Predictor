# Using the data that was scraped in scraper.py train a Random Forest model using scikit-learn 
#Model will be used to predict the outcome of games from 1/1/24 unitl the end of the season and determine the accuracy and precision


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
#Using random forest because it can capture non-linear relationships in the data
    #Opponent code of 15 doesn't mean that they are the 15th hardest opponent to play etc.

games = pd.read_csv("prem_matches.csv", index_col=0) #Read the df and make the first column the index column

#Clean the data
#ML can only work with numeric data taypes so need to convert non numeric data types

games["Date"] = pd.to_datetime(games["Date"])

games[r"Home\Away"] = games["Venue"].astype("category").cat.codes     #Set the venue to be a 1 (home game) or a 0 (away game)
games["Opponent Code"] = games["Opponent"].astype("category").cat.codes #Assign each team their own code
games["Hour"] = games["Time"].str.replace(":.+", "", regex=True).astype(int)    #Some teams may play better at certain parts of the day
games["Day Code"] = games["Date"].dt.day_of_week #Some teams may play better during certain days of the week

#Set target as whether the team won or lost
games["Target"] = (games["Result"] == "W").astype("int") #Set a win to be a 1 and a draw or loss to be a 0


forest = RandomForestClassifier(n_estimators=4200, min_samples_split=70, random_state=1) #Can adjust numbers to better fit model 
# n_estimators: number of decesion trees being trained, higher the number -> usually more accurate 
# min_samples_split: number of samples in a leaf of the decision tree, higher number -> less likley to overfit but lower accuracy on training data
# random_state: Controls the randomness of the estimator. Eensures that the random decisions made by the algorithm are reproducible across multiple runs of the same code

#Make sure all the training data comes before the testing data
training_data = games[games["Date"] < '2024-01-01'] #Any game before 2024
test_data = games[games["Date"] > '2024-01-01'] #Any game in 2024

predictors = [r"Home\Away", "Opponent Code", "Hour", "Day Code"]

forest.fit(training_data[predictors], training_data["Target"]) #train a RF model with the predictors trying to predict if the team won or lost

predictions = forest.predict(test_data[predictors]) #Make predictions

#Determine accuracy of the model

accuracy = accuracy_score(test_data["Target"], predictions) #0.62295 not bad accuracy

combined_df = pd.DataFrame(dict(Actual=test_data["Target"], Prediction=predictions))
df = pd.crosstab(index=combined_df["Actual"], columns=combined_df["Prediction"])
'''
            Prediction
                0   1
    Actual  0  168  61
            1  77  60
Did not predict wins very well but was good with predicting losses/draws so time to revise the algorithm to predict wins better
'''

p = precision_score(test_data["Target"], predictions) #0.49586 not too good precision

#Improve precision with rolling averages
#Factor in a teams form into the prediction
# group: Dataset to find rolling averages on
# cols: find rollings avgs on these columns
# new_cols: assign rolling averages to these columns
def rolling_avgs(group, cols, new_cols):
    group = group.sort_values("Date")
    rollings_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rollings_stats
    group = group.dropna(subset=new_cols) #Drop missings values, ex) Trying to get the previous 3 weeks in weeks 1 or 2
    return group


# Define columns and their new names for rolling averages
cols = ["xG", "xGA", "GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
new_cols = [f"Rolling {c}" for c in cols]

games_rolling = games.groupby("Team").apply(lambda x: rolling_avgs(x, cols, new_cols)).droplevel('Team').reset_index(drop=True)

#Retrain ML model

def make_predictions(data, predictions):
    training_data = data[data["Date"] < '2024-01-01'] #Any game before 2024
    test_data = data[data["Date"] > '2024-01-01'] #Any game in 2024
    forest.fit(training_data[predictors], training_data["Target"]) #train a RF model with the predictors trying to predict if the team won or lost
    predictions = forest.predict(test_data[predictors]) #Make predictions
    combined_df = pd.DataFrame(dict(Actual=test_data["Target"], Prediction=predictions))
    prec = precision_score(test_data["Target"], predictions)
    return combined_df, prec


combined, precision = make_predictions(games_rolling, predictors + new_cols)

combined = combined.merge(games_rolling[["Date", "Team", "Opponent", "Result"]], left_index=True, right_index=True) #Add into combined the Team, Opponent, Date and Result 

#Clean team name to make them the same in both the "Team" and Opponent" columns
class MissingDict(dict): #Create class which inherits from dictionary class
    __missing__ = lambda self, key: key
map = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham", 
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map) #Create MissingDict instance

combined["New Team"] = combined["Team"].map(mapping)

merged = combined.merge(combined, left_on=["Date", "New Team"], right_on=["Date", "Opponent"]) # finding both the home and away team predictions and merging them 

merged[(merged["Prediction_x"] == 1) & (merged["Prediction_y"] == 0)]["Actual_x"].value_counts() #See what happens when the model predicts Team A to win and team B to lose
'''
1    40
0    29
40/29 = 0.5797 = 58% precision
'''


#inspired by dataquest tutorial 
