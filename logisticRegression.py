import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def getColumnsToRemove(train_data, test_data, feature):
    if len(set(train_data[feature].unique())) > len(set(test_data[feature].unique())):
        return set(train_data[feature].unique()).symmetric_difference(
            set(test_data[feature].unique()))
    else:
        print "let me think"


def cleanFeatures(train_data, test_data):
    # Generating label Encoders for each feature
    le = LabelEncoder()
    featuresToEncode = train_data[['SEX', 'PERSON_TYPE']]
    featuresToEncode = featuresToEncode.apply(lambda x: le.fit_transform(x))

    # encoding case_state feature
    # encoding sex feature
    sex_onehot_features = pd.get_dummies(train_data['SEX'])
    # encoding person type feature
    person_type_onehot_features = pd.get_dummies(train_data['PERSON_TYPE'])
    columnsToRemove = getColumnsToRemove(train_data, test_data, 'PERSON_TYPE')
    person_type_onehot_features = person_type_onehot_features.drop(
        columns=columnsToRemove)
    # encoding age
    age_features = train_data['AGE']

    data = pd.concat([sex_onehot_features, age_features],
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

# Load test data and label it
df_test = pd.read_csv('fars_test.out', delimiter=",")
df_test.columns = ["CASE_STATE", "AGE", "SEX", "PERSON_TYPE", "SEATING_POSITION", "RESTRAINT_SYSTEM-USE", "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH", "EXTRICATION", "NON_MOTORIST_LOCATION", "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE", "ALCOHOL_TEST_RESULT", "POLICE-REPORTED_DRUG_INVOLVEMENT",
                   "METHOD_OF_DRUG_DETERMINATION", "DRUG_TEST_TYPE", "DRUG_TEST_RESULTS_(1_of_3)", "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_RESULTS_(2_of_3)", "DRUG_TEST_TYPE_(3_of_3)", "DRUG_TEST_RESULTS_(3_of_3)", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL", "RELATED_FACTOR_(1)-PERSON_LEVEL", "RELATED_FACTOR_(2)-PERSON_LEVEL", "RELATED_FACTOR_(3)-PERSON_LEVEL", "RACE"]

# Get both label and features
label = df["INJURY_SEVERITY"]
features = df.drop(columns=['INJURY_SEVERITY'])
clean_features = cleanFeatures(features, df_test)

# clf = LogisticRegression(random_state=0, solver='lbfgs',
#                          multi_class='ovr')
clf = LogisticRegression()
# clf.fit(new_features, label)


# new_features = cleanFeatures(df_test)
# prediction = clf.predict(new_features).tolist()
