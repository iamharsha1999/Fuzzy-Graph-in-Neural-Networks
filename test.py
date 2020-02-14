from Fuzzy import Model


model = Model(10)
model.add_layer(5)
model.add_layer(10)
model.add_layer(15)
model.add_layer(5)

model.train_model()
