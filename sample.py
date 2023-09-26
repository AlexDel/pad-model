import torch
from torch.nn import MSELoss
from torch.optim import SGD

mse_loss = MSELoss()

weights = torch.tensor([[0.1125, 0.7760]], requires_grad=True)
bias = torch.tensor([[0.6158]], requires_grad=True)

optimizer = SGD([weights, bias], lr= 0.001)

teeth = torch.tensor([[0.9, 0.9]])
ball = torch.tensor([[0.1, 0.2]])

dataset = [
    (teeth, torch.tensor([[1.0]])),
    (ball, torch.tensor([[0.1]]))
]

def get_threat_score(tensor_object):
    return tensor_object @ weights.T  + bias

def calc_loss(pred, actual):
    return mse_loss(pred, actual)


for i in range(5):
    for x, y in dataset:
        optimizer.zero_grad()
        x = get_threat_score(x)
        print("Before update")

        loss = mse_loss(x, y)

        loss.backward()

        optimizer.step()

        print("After update")
        print(weights, bias)
