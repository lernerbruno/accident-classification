import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def cleanFeatures(data):

    # Generating label Encoders for each feature
    le = LabelEncoder()
    sex_labels = le.fit_transform(data['SEX'])
    # person_type_labels = le.fit_transform(data['PERSON_TYPE'])
    # age_labels = le.fit_transform(data['AGE'])

    sex_onehot_features = pd.get_dummies(data['SEX'])
    # person_type_onehot_features = pd.get_dummies(data['PERSON_TYPE'])
    # age_onehot_features = pd.get_dummies(data['AGE'])

    data = pd.concat([sex_onehot_features],
                     axis=1)
    print data
    return data

    # print pd.concat([data[['PERSON_TYPE', 'SEX']], person_type_onehot_features],
    #                 axis=1).iloc[4:10]
    # training_dataframe = pd.concat([data[['PERSON_TYPE', 'SEX']], person_type_onehot_features],
    #                                axis=1)
    # training_dataframe = pd.DataFrame(
    #     {'person_type': list(person_type_onehot_features.values.flatten()), 'sex': list(sex_onehot_features.values.flatten())})
    # print training_dataframe[['person_type', 'sex']]


# Load training data and label it
df = pd.read_csv('fars_train.out', delimiter=",")

df.columns = ["CASE_STATE", "AGE", "SEX", "PERSON_TYPE", "SEATING_POSITION", "RESTRAINT_SYSTEM-USE", "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH", "EXTRICATION", "NON_MOTORIST_LOCATION", "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE", "ALCOHOL_TEST_RESULT", "POLICE-REPORTED_DRUG_INVOLVEMENT",
              "METHOD_OF_DRUG_DETERMINATION", "DRUG_TEST_TYPE", "DRUG_TEST_RESULTS_(1_of_3)", "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_RESULTS_(2_of_3)", "DRUG_TEST_TYPE_(3_of_3)", "DRUG_TEST_RESULTS_(3_of_3)", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL", "RELATED_FACTOR_(1)-PERSON_LEVEL", "RELATED_FACTOR_(2)-PERSON_LEVEL", "RELATED_FACTOR_(3)-PERSON_LEVEL", "RACE", "INJURY_SEVERITY"]

# Get both label and features
label = df["INJURY_SEVERITY"]
features = df.drop(columns=['INJURY_SEVERITY'])
new_features = cleanFeatures(features)

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='ovr')
clf.fit(new_features, label)


# Load test data and label it
df_test = pd.read_csv('fars_test.out', delimiter=",")
df_test.columns = ["CASE_STATE", "AGE", "SEX", "PERSON_TYPE", "SEATING_POSITION", "RESTRAINT_SYSTEM-USE", "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH", "EXTRICATION", "NON_MOTORIST_LOCATION", "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE", "ALCOHOL_TEST_RESULT", "POLICE-REPORTED_DRUG_INVOLVEMENT",
                   "METHOD_OF_DRUG_DETERMINATION", "DRUG_TEST_TYPE", "DRUG_TEST_RESULTS_(1_of_3)", "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_RESULTS_(2_of_3)", "DRUG_TEST_TYPE_(3_of_3)", "DRUG_TEST_RESULTS_(3_of_3)", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL", "RELATED_FACTOR_(1)-PERSON_LEVEL", "RELATED_FACTOR_(2)-PERSON_LEVEL", "RELATED_FACTOR_(3)-PERSON_LEVEL", "RACE"]

new_features = cleanFeatures(df_test)
prediction = clf.predict(new_features).tolist()
print prediction


# clf.predict([["Alabama", "34", "Male", "Driver", "Front_Seat_-_Left_Side_(Drivers_Side)", "None_Used/Not_Applicable", "Air_Bag_Available_but_Not_Deployed_for_this_Seat", "Totally_Ejected", "Unknown", "Not_Extricated", "Not_Applicable_-_Vehicle_Occupant", "Yes_(Alcohol_Involved)", "Not_Reported", "Whole_Blood", 97, "Reported_Unknown", "Not_Reported", "Unknown_if_Tested_for_Drugs", 999, "Not_Tested_for_Drugs", 000, "Not_Tested_for_Drugs", 000, "Non-Hispanic", "No", "Not_Applicable_-_Driver/None_-_All_Other_Persons,Not_Applicable_-_Driver/None_-_All_Other_Persons", "Not_Applicable_-_Driver/None_-_All_Other_Persons", "White"]]).tolist())
