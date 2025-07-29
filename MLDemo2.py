#Import necessari libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Loading the Dataset and Displaying the First Few Rows
iris_data = pd.read_csv('iris.csv')
iris_data.head()

#Split data into features (X) and labels (y)
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

#Create a ML Model
model = LogisticRegression()

#Training the model
model.fit(X.values, y)

#Predict using the trained model
predictions = model.predict([[4.6, 3.5, 1.5, 0.2]])

#Print the predictions
print(predictions)
