import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer


def cleanFeatures(data):
    data['PERSON_TYPE'] = np.where(
        data['PERSON_TYPE'] == 'Other_Pedestrian', 'Pedestrian', data['PERSON_TYPE'])
    data['PERSON_TYPE'] = np.where(
        data['PERSON_TYPE'] == 'Other_Cyclist', 'Bicyclist', data['PERSON_TYPE'])

    print data["PERSON_TYPE"].unique()


# Load training data and label it
df = pd.read_csv('fars_train.out', delimiter=",")

df.columns = ["CASE_STATE", "AGE", "SEX", "PERSON_TYPE", "SEATING_POSITION", "RESTRAINT_SYSTEM-USE", "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH", "EXTRICATION", "NON_MOTORIST_LOCATION", "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE", "ALCOHOL_TEST_RESULT", "POLICE-REPORTED_DRUG_INVOLVEMENT",
              "METHOD_OF_DRUG_DETERMINATION", "DRUG_TEST_TYPE", "DRUG_TEST_RESULTS_(1_of_3)", "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_RESULTS_(2_of_3)", "DRUG_TEST_TYPE_(3_of_3)", "DRUG_TEST_RESULTS_(3_of_3)", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL", "RELATED_FACTOR_(1)-PERSON_LEVEL", "RELATED_FACTOR_(2)-PERSON_LEVEL", "RELATED_FACTOR_(3)-PERSON_LEVEL", "RACE", "INJURY_SEVERITY"]
# print MultiLabelBinarizer().fit_transform(df)
# Get both label and features
label = df["INJURY_SEVERITY"]
# get features and remove unrelated data like HISPANIC_ORIGIN, TAKEN_TO_HOSPITAL(?)
features = df.drop(columns=['HISPANIC_ORIGIN', 'INJURY_SEVERITY', "SEX"])
cleanFeatures(features)

# print features

# Load test data and label it
df_test = pd.read_csv('fars_test.out', delimiter=",")
df_test.columns = ["CASE_STATE", "AGE", "SEX", "PERSON_TYPE", "SEATING_POSITION", "RESTRAINT_SYSTEM-USE", "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH", "EXTRICATION", "NON_MOTORIST_LOCATION", "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE", "ALCOHOL_TEST_RESULT", "POLICE-REPORTED_DRUG_INVOLVEMENT",
                   "METHOD_OF_DRUG_DETERMINATION", "DRUG_TEST_TYPE", "DRUG_TEST_RESULTS_(1_of_3)", "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_RESULTS_(2_of_3)", "DRUG_TEST_TYPE_(3_of_3)", "DRUG_TEST_RESULTS_(3_of_3)", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL", "RELATED_FACTOR_(1)-PERSON_LEVEL", "RELATED_FACTOR_(2)-PERSON_LEVEL", "RELATED_FACTOR_(3)-PERSON_LEVEL", "RACE"]
# Remove unnecessary data
df_test = df_test.drop(columns=['HISPANIC_ORIGIN'])

# regr = linear_model.LinearRegression()
# regr.fit(features, label)
# print regr.predict([["Alabama","34","Male","Driver","Front_Seat_-_Left_Side_(Drivers_Side)","None_Used/Not_Applicable","Air_Bag_Available_but_Not_Deployed_for_this_Seat","Totally_Ejected","Unknown","Not_Extricated","Not_Applicable_-_Vehicle_Occupant","Yes_(Alcohol_Involved)","Not_Reported","Whole_Blood",97,"Reported_Unknown","Not_Reported","Unknown_if_Tested_for_Drugs",999,"Not_Tested_for_Drugs",000,"Not_Tested_for_Drugs",000,"Non-Hispanic","No","Not_Applicable_-_Driver/None_-_All_Other_Persons,Not_Applicable_-_Driver/None_-_All_Other_Persons","Not_Applicable_-_Driver/None_-_All_Other_Persons","White"]]).tolist()

le = preprocessing.LabelEncoder()


clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='ovr')
le.fit(label)
# clf.fit(features, label)

# print clf

# clf.predict([["Alabama", "34", "Male", "Driver", "Front_Seat_-_Left_Side_(Drivers_Side)", "None_Used/Not_Applicable", "Air_Bag_Available_but_Not_Deployed_for_this_Seat", "Totally_Ejected", "Unknown", "Not_Extricated", "Not_Applicable_-_Vehicle_Occupant", "Yes_(Alcohol_Involved)", "Not_Reported", "Whole_Blood", 97, "Reported_Unknown", "Not_Reported", "Unknown_if_Tested_for_Drugs", 999, "Not_Tested_for_Drugs", 000, "Not_Tested_for_Drugs", 000, "Non-Hispanic", "No", "Not_Applicable_-_Driver/None_-_All_Other_Persons,Not_Applicable_-_Driver/None_-_All_Other_Persons", "Not_Applicable_-_Driver/None_-_All_Other_Persons", "White"]]).tolist())
