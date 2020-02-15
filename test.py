from Fuzzy import Model


model = Model(10)
model.add_layer(5, "AND")
model.add_layer(10, "AND")
model.add_layer(15, "AND")
model.add_layer(5, "AND")

model.train_model()
