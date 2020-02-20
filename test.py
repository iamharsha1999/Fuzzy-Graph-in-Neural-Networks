from Fuzzy import Model
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

csv_path = '/home/harsha/Downloads/hotel_bookings.csv'
df = pd.read_csv(csv_path, delimiter=',')
df.drop(['lead_time','agent','company','adr','reservation_status_date'], axis = 1, inplace = True)
df.dropna(inplace=True)

le = LabelEncoder()
df = df.apply(le.fit_transform)

print("DataSet")
print(df.head())
x = df.iloc[0:10,:-1]
y = df.iloc[0:10,-1]

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.3)
x_train = np.random.randn(2,3)
y_train = np.random.randn(2,1)
# print(x_train.iloc[0,:])
# print(y_train.iloc[0])


model = Model(3)
model.add_layer(3, "AND")
model.add_layer(3, "AND")
# model.add_layer(10, "AND")
# model.add_layer(15, "AND")
# model.add_layer(1, "AND")
model.train_model(x_train,y_train)
