# ITC Challenge - Bruno Lerner
import pandas as pd
import numpy as np

# This file was made to compare the result of the prediction with 80% of training dataset
# with the other 20 % used to predict

# getting real values
df = pd.read_csv('fars_train.out', delimiter=",").iloc[5400:]
df.columns = ["CASE_STATE", "AGE", "SEX", "PERSON_TYPE", "SEATING_POSITION", "RESTRAINT_SYSTEM-USE", "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH", "EXTRICATION", "NON_MOTORIST_LOCATION", "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE", "ALCOHOL_TEST_RESULT", "POLICE-REPORTED_DRUG_INVOLVEMENT",
              "METHOD_OF_DRUG_DETERMINATION", "DRUG_TEST_TYPE", "DRUG_TEST_RESULTS_(1_of_3)", "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_RESULTS_(2_of_3)", "DRUG_TEST_TYPE_(3_of_3)", "DRUG_TEST_RESULTS_(3_of_3)", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL", "RELATED_FACTOR_(1)-PERSON_LEVEL", "RELATED_FACTOR_(2)-PERSON_LEVEL", "RELATED_FACTOR_(3)-PERSON_LEVEL", "RACE", "INJURY_SEVERITY"]

label = df['INJURY_SEVERITY'].values

# getting predicted values
prediction_file = open("predictionTest.txt", "r")
wins = 0
loss = 0
i = 0

for prediction in prediction_file.read().split(","):
    if prediction == label[i]:
        wins += 1
    else:
        loss += 1
    i += 1
print "dataset length: " + str(i)
print "number of match: " + str(wins)
print "number of non-match: " + str(loss)
print "percentage: " + str(float(wins)/float(loss+wins))
