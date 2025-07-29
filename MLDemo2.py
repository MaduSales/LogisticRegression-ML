#Import necessari libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Loading the Dataset and Displaying the First Few Rows
iris_data = pd.read_csv('iris.csv')
iris_data.head()

#Split data into features (X) and labels (y)
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']
