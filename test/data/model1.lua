require 'nn'

mod = {}

mod['linear-bias'] = nn.Linear(20, 10):float()
mod['linear-nobias'] = nn.Linear(20, 10, false):float()

torch.save("model1.t7", mod)