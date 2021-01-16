import pandas as pd
import numpy as np


exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'], 'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19], 'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1], 'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

dataFrame = pd.DataFrame(exam_data, labels)
print(dataFrame)

#b. i.replacing qualify column attributes yes or no with True or False
dataFrame.loc[(dataFrame.qualify == 'yes'), ('qualify')] = True
dataFrame.loc[(dataFrame.qualify == 'no'), ('qualify')] = False
#print(dataFrame)

#printing data frames which are not null
#dataFrameQualified = dataFrame[dataFrame.score.notnull()]
#print(dataFrameQualified)

#b. ii.printing data frame in attempts, name, qualify, score column format
dataFrameModified = dataFrame[['attempts', 'name', 'qualify', 'score']]
print(dataFrameModified)





