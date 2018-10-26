import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def getColumnsToRemove(train_data, test_data, feature):
    return set(train_data[feature].unique()).symmetric_difference(
        set(test_data[feature].unique()))


def normalizeData(columnsToRemove, feature, test_feature):
    for column in columnsToRemove:
        if column not in feature:
            feature[column] = pd.Series(
                0, index=feature.index)
        if column not in test_feature:
            test_feature[column] = pd.Series(
                0, index=test_feature.index)
    return feature, test_feature


def transformFeatures(train_data, test_data):
    # Generating label Encoders for each feature
    le = LabelEncoder()
    featuresToEncode = train_data[['CASE_STATE', 'SEX', 'PERSON_TYPE']]
    featuresToEncode = featuresToEncode.apply(lambda x: le.fit_transform(x))

    # adding age
    age_features = train_data['AGE']
    age_features_test = test_data['AGE']
    data = pd.concat([age_features],
                     axis=1)
    data_test = pd.concat([age_features_test],
                          axis=1)

    for featureToEncode in featuresToEncode:
        onehot_feature = pd.get_dummies(train_data[featureToEncode])
        onehot_feature_test = pd.get_dummies(train_data[featureToEncode])
        columnsToRemove = getColumnsToRemove(
            train_data, test_data, featureToEncode)
        onehot_feature, onehot_feature_test = normalizeData(columnsToRemove, onehot_feature,
                                                            onehot_feature_test)
        data = data.join(onehot_feature)
        data_test = data_test.join(onehot_feature_test)

    return data, data_test


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
training_features, test_features = transformFeatures(features, df_test)
print test_features
print training_features
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='ovr')
clf.fit(training_features, label)


prediction = clf.predict(test_features).tolist()
# print prediction
