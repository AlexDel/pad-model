from torch import Tensor
from torch.nn import MSELoss

mse_loss = MSELoss()

weights = Tensor([[0.1125, 0.7760]])
bias = Tensor([[0.6158]])

teeth = Tensor([[0.9, 0.9]])
ball = Tensor([[0.1, 0.2]])

dataset = [
    (teeth, 1),
    (ball, 0)
]

def get_threat_score(tensor_object):
    return tensor_object @ weights.T  + bias

def calc_loss(pred, actual):
    return mse_loss(pred, actual)


score = get_threat_score(ball)
ball_act = Tensor([[0]])

loss = calc_loss(score, ball_act)
loss
