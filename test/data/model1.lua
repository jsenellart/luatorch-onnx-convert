require 'nn'

local mod = {}

mod['linear-bias'] = nn.Linear(20, 10)
mod['linear-nobias'] = nn.Linear(20, 10, false)
mod['cadd-table'] = nn.CAddTable()

torch.save("model1.t7", mod)