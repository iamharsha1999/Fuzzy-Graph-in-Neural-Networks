from Fuzzy import Model
import torch
from skfuzzy.membership import trimf


a = torch.tensor([1,2,3,4,5,6,7,8,9,10])
print(a)
a = torch.from_numpy(trimf(a,[0,7,10]))
a = a.to(device = torch.device('cuda')).float()
a = a.view(-1,1)
o = torch.randn(5,1, device = 'cuda:0')

model = Model(10)
model.add_layer(5, "AND")
model.add_layer(10, "AND")
model.add_layer(15, "AND")
model.add_layer(5, "AND")

model.train_model(a,o)
