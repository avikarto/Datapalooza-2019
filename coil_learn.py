import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#
cwd = os.getcwd()
print(cwd)
df = pd.read_csv(cwd + '/cleanedDatafiles/coil_clean.csv')
# We have data for years 2011-2017.  11-17 are possible values for "year".
year = '14'

# %%

len(df)

# %%
# Drop rows where the average age is out of bounds for whole life insurance
#       Only ages 45-85 are admissable - These values correspond to selecting
#       categories 3-6 of the map L1
# Dropping rows where the L1 category is 1 or 2...

labs = [i for i in range(len(df)) if df['4 MGEMLEEF Avg age see L1'][i] <= 2]
df.drop(axis='index', labels=labs, inplace=True)

print(len(labs), 'elements were dropped.')

# Dropping a duplicate index column
df.drop(columns=['Unnamed: 0'], inplace=True)
df.head(1)

# %%
# Default column names are cancer.
df.columns = ['dropThis2', 'dropThis3', '4_avgAge_L1', '10_married',\
              '11_livingTogether', '12_otherRelation', '13_singles',\
              '14_noChildren', '15_withChildren', '16_highEdu', '17_medEdu',\
              '18_lowEdu', '30_renters', '31_owners', '32_1car', '33_2car',\
              '34_0car', 'dropThis35', '36_privateIns', '37_incomeTo30k',\
              '38_incomeTo45k', '39_incomeTo75k', '40_incomeTo122k',\
              '41_incomeMax', 'dropThis42', 'dropThis43', '76_nLifeIns']
# drop more columns where data isn't helpful or well-described
df.drop(columns=['dropThis2', 'dropThis3', 'dropThis35', 'dropThis42', 'dropThis43'], inplace=True)
df.set_index(np.array(range(len(df))), inplace=True)

df.head(1)


# %%
# to train on percentages of the population, we first need to determine the population
# This data doesn't seem consistent, so I'll try to find a sensable population
df['numPpl1'] = df['16_highEdu'] + df['17_medEdu'] + df['18_lowEdu']
df['numPpl2'] = df['37_incomeTo30k'] + df['38_incomeTo45k'] + df['39_incomeTo75k'] +\
    df['40_incomeTo122k'] + df['41_incomeMax']
df['numPpl3'] = df['30_renters'] + df['31_owners']
df['numPpl4'] = df['14_noChildren'] + df['15_withChildren']
df['numPpl5'] = df['10_married'] + df['11_livingTogether'] + df['12_otherRelation'] + df['13_singles']

temp = pd.DataFrame(data=[df['numPpl1'], df['numPpl2'], df['numPpl3'], df['numPpl4'], df['numPpl5']], copy=True)
pplDF = temp.T
pplDF.set_index(np.array(range(len(pplDF))), inplace=True)
pplDF['avgPpl'] = (pplDF['numPpl1']+pplDF['numPpl2']+pplDF['numPpl3']+pplDF['numPpl4']+pplDF['numPpl5'])/5.0
pplDF.head(3)

# %%
# generate the usable features frame - to train the model
featuresDF = df.copy(deep=True)
featuresDF.drop(columns=['numPpl1', 'numPpl2', 'numPpl3', 'numPpl4', 'numPpl5'], inplace=True)
featuresDF.set_index(np.array(range(len(featuresDF))), inplace=True)
featuresDF.head()

# %%
# generate the independent probabilities of desired factors - for training
probDF = pd.DataFrame()
probDF['ageGroup'] = featuresDF['4_avgAge_L1']
probDF['P_homeOwner'] = featuresDF['31_owners'] / pplDF['avgPpl']
probDF['P_ins'] = featuresDF['76_nLifeIns'] / pplDF['avgPpl']
# the calculated average number of people per area is sometimes less than a feature count.  Catch this.
if probDF['P_homeOwner'].values.max() > 1:
    probDF['P_homeOwner'] /= probDF['P_homeOwner'].values.max()
if probDF['P_ins'].values.max() > 1:
    probDF['P_ins'] /= probDF['P_ins'].values.max()
# computing a boolean insurance purchase flag from the high-tail of prob.
threshold = np.mean(probDF['P_ins']) + 1.0*np.std(probDF['P_ins'])
probDF['bool_ins'] = np.floor(probDF['P_ins'] + (1-threshold))
probDF.set_index(np.array(range(len(probDF))), inplace=True)
probDF.head()

# %%
# Educational demographic info by zip code - to be predicted from
eduDF = pd.read_csv(cwd+'/cleanedDatafiles/ACS_'+year+'_5YR_S1501_with_ann_cleaned.csv')
eduDF.drop(columns=['Unnamed: 0'], inplace=True)
eduDF['totalPpl'] = eduDF['45to65Total'] + eduDF['Over65Total']
eduDF['totalBachelors'] = eduDF['45to65Bachelors'] + eduDF['Over65Bachelors']
eduDF['totalHighschool'] = eduDF['45to65Highschool'] + eduDF['Over65Highschool']
eduDF['totalLow'] = eduDF['totalPpl'] - eduDF['totalBachelors'] - eduDF['totalHighschool']
# some of the data is absurd - the number of people in an age group with a terminal BS plus those
# with a terminal HS education exceed the total people for that age bracket.  This difference would
# otherwise indicate a count of "low" education, less than a HS diploma.  In such cases, this feature
# is set to zero.
eduDF['totalLow'][eduDF['totalLow'] < 0] = 0
# calculating proportions per age group per education bracket
eduDF['P_Low'] = eduDF['totalLow'] / eduDF['totalPpl']
eduDF['P_HS'] = eduDF['totalHighschool'] / eduDF['totalPpl']
eduDF['P_Bachelors'] = eduDF['totalBachelors'] / eduDF['totalPpl']
# sometimes these probabilities can be NaN, since the total number of people is repored as zero.
eduDF['P_Low'][eduDF['totalPpl'] == 0] = 0
eduDF['P_HS'][eduDF['totalPpl'] == 0] = 0
eduDF['P_Bachelors'][eduDF['totalPpl'] == 0] = 0
# reindex to align with other DFs
eduDF.set_index(np.array(range(len(eduDF))), inplace=True)
eduDF.head()

# %%
eduDF.isnull().values.any()

# %%
# Age demographic info by zip code - to be predicted from
ageDF = pd.read_csv(cwd+'/cleanedDatafiles/ACS_'+year+'_5YR_S0101_with_ann_cleaned.csv')
ageDF.drop(columns=['Unnamed: 0'], inplace=True)
for col in ageDF.columns:
    if col not in ['ZCTA', 'TotalPopulation']:
        ageDF[col] /= 100.
ageDF['avgAge'] = (50*ageDF['45to54'] + 60*ageDF['55to64'] + 70*ageDF['65to74'] + 80*ageDF['75to84']) / \
    (ageDF['45to54'] + ageDF['55to64'] + ageDF['65to74'] + ageDF['75to84'] + ageDF['85andOver'])
# Fill any NaN values in avgAge with the mean of the column
ageDF['avgAge'].fillna(value=np.mean(ageDF['avgAge']), inplace=True)
# convert average ages to COIL L1 configuration, and divide by 10 for feature scaling
ageDF['avgAgeScaled'] = np.floor((ageDF['avgAge'] / 10 - 1)) / 10  # -1 since, for instance, 50-60 maps to 4 in L1
ageDF.head()

# %%
ageDF['avgAge'].isnull().values.any()

# %%
# Income demographic info by zip code - to be predicted from
incomeDF = pd.read_csv(cwd+'/cleanedDatafiles/ACS_'+year+'_5YR_S2503_with_ann_cleaned.csv')
incomeDF.drop(columns=['Unnamed: 0'], inplace=True)
# Will sucks at giving columns useful names
incomeDF.columns = ['ZCTA', 'below35k', '35to50k', '50to75k', 'above75k']
incomeDF['totalPercent'] = incomeDF['below35k'] + incomeDF['35to50k'] + incomeDF['50to75k'] + incomeDF['above75k']
# the total percentages don't add to 100.  Distribute the difference into the other categories evenly.
incomeDF['diff'] = 100.0 - incomeDF['totalPercent']
for col in incomeDF.columns:
    if col not in ['ZCTA', 'totalPercent', 'diff']:
        incomeDF[col] += incomeDF['diff'] / 4.0
        incomeDF[col] /= 100.0  # feature scaling
incomeDF['totalPercent'] = incomeDF['below35k'] + incomeDF['35to50k'] + incomeDF['50to75k'] + incomeDF['above75k']
incomeDF['diff'] = 1.0 - incomeDF['totalPercent']
incomeDF.head()

# %%
incomeDF.isnull().values.any()

# %%
# Marriage demographic info by zip code - to be predicted from
marriageDF = pd.read_csv(cwd+'/cleanedDatafiles/ACS_'+year+'_5YR_S1201_with_ann_cleaned.csv')
marriageDF.drop(columns=['Unnamed: 0'], inplace=True)
marriageDF.head()

# %%
marriageDF.isnull().values.any()

# %%
# Car ownership demographic info by zip code - to be predicted from
carDF = pd.read_csv(cwd+'/cleanedDatafiles/ACS_'+year+'_5YR_S0802_with_ann_cleaned.csv')
# drop the columns we don't need...
for col in carDF.columns:
    if col not in ['ZCTA', 'TotalNoVehicle', 'Total1Vehicle', 'Total2Vehicle', 'Total3orMoreVehicle']:
        carDF.drop(columns=[col], inplace=True)
# COIL has the following relevant features: '32_1car', '33_2car', '34_0car'
carDF['Total2orMoreVehicle'] = carDF['Total2Vehicle'] + carDF['Total3orMoreVehicle']
carDF.drop(columns=['Total2Vehicle', 'Total3orMoreVehicle'], inplace=True)
carDF['totalPercent'] = carDF['TotalNoVehicle'] + carDF['Total1Vehicle'] + carDF['Total2orMoreVehicle']
# the total percentages don't always add to 100.  Distribute the difference into the other categories evenly.
carDF['diff'] = 100.0 - carDF['totalPercent']
for col in carDF.columns:
    if col not in ['ZCTA', 'totalPercent', 'diff']:
        carDF[col] += carDF['diff'] / 3.0
        carDF[col] /= 100.0  # feature scaling
carDF.drop(columns=['diff', 'totalPercent'], inplace=True)
carDF.head()

# %%
marriageDF.isnull().values.any()

# %%
# import home owner data by zip - to be predicted from
homeOwnerDF = pd.read_csv(cwd+'/cleanedDatafiles/ACS_'+year+'_5YR_S1101_with_ann_cleaned.csv')
homeOwnerDF.drop(columns=['Unnamed: 0', 'RENT'], inplace=True)
homeOwnerDF['OWN'] /= 100.
homeOwnerDF.head()

# %%
homeOwnerDF.isnull().values.any()

# %%
# generating traing vectors
xTrain = pd.DataFrame()
xTrain['16_highEdu'] = df['16_highEdu'] / pplDF['avgPpl']
xTrain['17_medEdu'] = df['17_medEdu'] / pplDF['avgPpl']
# xTrain['18_lowEdu'] = df['18_lowEdu'] / pplDF['avgPpl']  # overspecified
xTrain['avgAgeGroupScaled'] = df['4_avgAge_L1'] / 10
xTrain['lowIncome'] = df['37_incomeTo30k'] / pplDF['avgPpl']
xTrain['medLowIncome'] = df['38_incomeTo45k'] / pplDF['avgPpl']
xTrain['medHighIncome'] = df['39_incomeTo75k'] / pplDF['avgPpl']
# xTrain['highIncome'] = (df['40_incomeTo122k'] + df['41_incomeMax']) / pplDF['avgPpl']  # overspecified
xTrain['10_married'] = df['10_married'] / pplDF['avgPpl']
xTrain['34_0car'] = df['34_0car'] / pplDF['avgPpl']
xTrain['32_1car'] = df['32_1car'] / pplDF['avgPpl']
# xTrain['33_2car'] = df['33_2car'] / pplDF['avgPpl']  # overspecified
xTrain['P_homeOwner'] = probDF['P_homeOwner'] / pplDF['avgPpl']
# owing to the method of caluating avgPpl in pplDF, some of the probabilities here are > 1. Scale these columns down.
for col in xTrain.columns:
    if xTrain[col].max() > 1:
        xTrain[col] /= xTrain[col].max()
xTrain.set_index(np.array(range(len(xTrain))), inplace=True)
xTrain.head(3)

# %%
yTrain = probDF['P_ins']
yTrain.head(3)

# %%
# are the relations linear?
# for col in xTrain.columns:
#     plt.scatter(xTrain[col],probDF['P_ins'])
#     plt.ylim(0,1)
#     plt.ylabel('P_ins')
#     plt.xlabel(col)
#     plt.show()
#
# TLDR: the COIL data is far too sparse to extract any truly meaningful scaling, as expected.
# Unfortunately, it's all we have to train with.  Assume linear scaling.

# %%
# How are the training parameters correlated with the target?
corrMat = xTrain.join(probDF['P_ins'])
corrMat.corr()['P_ins']

# %%
# importing the models
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
# from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
# from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC

# mnb = MultinomialNB()
# cnb = ComplementNB()
linModel = LinearRegression(fit_intercept=False)
tsModel = TheilSenRegressor(fit_intercept=False)
hrModel = HuberRegressor(fit_intercept=False)
# bardModel = ARDRegression()
brModel = BayesianRidge()
# enModel = ElasticNet()
ridgeModel = Ridge()
# logModel = LogisticRegression()
# rfModel = RandomForestClassifier()
# svcModel = SVC()

# %%
# cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size=0.2)
kf = KFold(n_splits=5)
linScores = []
tsScores = []
hrScores = []
# bardScores = []
brScores = []
# enScores = []
ridgeScores = []
for trainingSplits, testingSplits in kf.split(xTrain):
    x_train, x_test, y_train, y_test = xTrain.loc[trainingSplits], xTrain.loc[testingSplits], yTrain.loc[trainingSplits], yTrain.loc[testingSplits]
    linModel.fit(x_train, y_train)
    linScores.append(linModel.score(x_test, y_test))
    tsModel.fit(x_train, y_train)
    tsScores.append(tsModel.score(x_test, y_test))
    hrModel.fit(x_train, y_train)
    hrScores.append(hrModel.score(x_test, y_test))
    # bardModel.fit(x_train, y_train)
    # bardScores.append(bardModel.score(x_test, y_test))
    brModel.fit(x_train, y_train)
    brScores.append(brModel.score(x_test, y_test))
    # enModel.fit(x_train, y_train)
    # enScores.append(enModel.score(x_test, y_test))
    ridgeModel.fit(x_train, y_train)
    ridgeScores.append(ridgeModel.score(x_test, y_test))
print(linScores, np.mean(linScores))
print(tsScores, np.mean(tsScores))
print(hrScores, np.mean(hrScores))
# print(bardScores, np.mean(bardScores))
print(brScores, np.mean(brScores))
# print(enScores, np.mean(enScores))
print(ridgeScores, np.mean(ridgeScores))
# again, the results are sad.  Sparse COIL data for training.

# %%
# training the models
linModel.fit(xTrain, yTrain)
tsModel.fit(xTrain, yTrain)
hrModel.fit(xTrain, yTrain)
# bardModel.fit(xTrain, yTrain)
brModel.fit(xTrain, yTrain)
# enModel.fit(xTrain, yTrain)
ridgeModel.fit(xTrain, yTrain)
# logModel.fit(xTrain, probDF['bool_ins'])

# %%
# generate vectors to be predicted from
xValues = pd.DataFrame()
# xValues['P_homeOwner'] = homeOwnerDF['OWN'] / 100.
xValues['P_highEdu'] = eduDF['P_Bachelors']
xValues['P_medEdu'] = eduDF['P_HS']
# xValues['P_lowEdu'] = eduDF['P_Low']  # overspecified
xValues['avgAgeScaled'] = ageDF['avgAgeScaled']
xValues['lowIncome'] = incomeDF['below35k']
xValues['medLowIncome'] = incomeDF['35to50k']
xValues['medHighIncome'] = incomeDF['50to75k']
# xValues['highIncome'] = incomeDF['above75k']  # overspecified
xValues['P_married'] = marriageDF['marriedPercent']
xValues['P_noCar'] = carDF['TotalNoVehicle']
xValues['P_1Car'] = carDF['Total1Vehicle']
# xValues['P_2+Car'] = carDF['Total2orMoreVehicle']  # overspecified
xValues['P_homeOwner'] = homeOwnerDF['OWN']
# make sure all feature vectors are the same length
# for col in xValues.columns:
#     print(xValues[col].shape)
xValues.head(1)

# %%
# predict values
linPredictions = linModel.predict(xValues)
tsPredictions = tsModel.predict(xValues)
hrPredictions = hrModel.predict(xValues)
# bardPredictions = bardModel.predict(xValues)
brPredictions = brModel.predict(xValues)
# enPredictions = enModel.predict(xValues)
ridgePredictions = ridgeModel.predict(xValues)
# logPredictions = logModel.predict(xValues)
print('Features:')
print(xTrain.columns.values, '\n')
print("Linear coefficients:", '\n', linModel.coef_, '...Intercept:', linModel.intercept_, '\n')
print("TS coefficients:", '\n', tsModel.coef_, '...Intercept:', tsModel.intercept_, '\n')
print("HR coefficients:", '\n', hrModel.coef_, '...Intercept:', hrModel.intercept_, '\n')
# print("BARD coefficients:", '\n', bardModel.coef_, '...Intercept:', bardModel.intercept_, '\n')
print("BR coefficients:", '\n', brModel.coef_, '...Intercept:', brModel.intercept_, '\n')
# print("EN coefficients:", '\n', enModel.coef_, '...Intercept:', enModel.intercept_, '\n')
print("Ridge coefficients:", '\n', ridgeModel.coef_, '...Intercept:', ridgeModel.intercept_, '\n')
# print("Logistic coefficients:", '\n', logModel.coef_, '...Intercept:', logModel.intercept_, '\n')

# %%
# normalize the score output to the range [0,1] for plotting
for result in [linPredictions, tsPredictions, hrPredictions, brPredictions, ridgePredictions]:
    result -= result.min()
    result /= result.max()

# %%
# score histograms
binCount = 125
plt.title('Scores: Linear Model')
plt.hist(linPredictions, bins=binCount)
plt.show()
plt.title('Scores: TS Model')
plt.hist(tsPredictions, bins=binCount)
plt.show()
plt.title('Scores: HR Model')
plt.hist(hrPredictions, bins=binCount)
plt.show()
# plt.title('Scores: BARD Model')
# plt.hist(bardPredictions, bins=binCount)
# plt.show()
plt.title('Scores: BR Model')
plt.hist(brPredictions, bins=binCount)
plt.show()
# plt.title('Scores: EN Model')
# plt.hist(enPredictions, bins=binCount)
# plt.show()
plt.title('Scores: Ridge Model')
plt.hist(ridgePredictions, bins=binCount)
plt.show()
# plt.title('Scores: Logistic Model')
# plt.hist(logPredictions)
# plt.show()

# %%
# prepare the csv file for output to the plotter
linOutput = pd.DataFrame()
tsOutput = pd.DataFrame()
brOutput = pd.DataFrame()
for model in [linOutput, tsOutput, brOutput]:
    model['ZCTA'] = incomeDF['ZCTA']
# linOutput['Score'] = linPredictions
# tsOutput['Score'] = tsPredictions
brOutput['Score'] = brPredictions
# export the scores
# linOutput.to_csv(cwd+'/results/linScores'+year+'.csv')
# tsOutput.to_csv(cwd+'/results/tsScores'+year+'.csv')
brOutput.to_csv(cwd+'/results/brScores'+year+'.csv')

# %%
# brOutput[brOutput['Score']>0.6]


#


# %%
# if youWannaBeMyLover:
#     print('You gotta get with my friends')


#


# %%
pass
# %%
pass
# %%
pass
