require 'nn'

local mod = {}

mod['linear-bias'] = nn.Linear(20, 10)
mod['linear-bias']:forward(torch.randn(20))

mod['linear-nobias'] = nn.Linear(20, 10, false)
mod['linear-nobias']:forward(torch.randn(4,20))

mod['cadd-table'] = nn.CAddTable()
mod['cadd-table']:forward({torch.randn(3), torch.randn(3)})

mod['abs'] = nn.Abs()
mod['abs']:forward(torch.randn(15))

mod['tanh'] = nn.Tanh()
mod['tanh']:forward(torch.randn(15))

mod['sigmoid'] = nn.Sigmoid()
mod['sigmoid']:forward(torch.randn(15))

mod['lookup'] = nn.LookupTable(20, 100)
mod['lookup']:forward((torch.rand(3):abs()*20):int())

mod['reshape'] = nn.Reshape(4, 3, 7)
mod['reshape']:forward(torch.rand(42, 2))

mod['splittable'] = nn.SplitTable(2)
mod['splittable']:forward(torch.rand(42, 2))

torch.save("model1.t7", mod)