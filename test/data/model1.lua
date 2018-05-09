require 'nn'

local mod = {}

mod['linear-bias'] = nn.Linear(20, 10)
mod['linear-nobias'] = nn.Linear(20, 10, false)
mod['cadd-table'] = nn.CAddTable()
mod['abs'] = nn.Abs()
mod['tanh'] = nn.Tanh()
mod['sigmoid'] = nn.Sigmoid()
mod['lookup'] = nn.LookupTable(100, 20)
mod['reshape'] = nn.Reshape(4, 3, 10)

torch.save("model1.t7", mod)