require 'nn'

local mod = {}

mod['linear-bias'] = nn.Linear(20, 10)
mod['linear-bias']:forward(torch.randn(7, 20))

mod['linear-nobias'] = nn.Linear(20, 10, false)
mod['linear-nobias']:forward(torch.randn(7, 20))

mod['cadd-table'] = nn.CAddTable()
mod['cadd-table']:forward({torch.randn(7, 3), torch.randn(7, 3)})

mod['abs'] = nn.Abs()
mod['abs']:forward(torch.randn(7, 15))

mod['tanh'] = nn.Tanh()
mod['tanh']:forward(torch.randn(7, 15))

mod['sigmoid'] = nn.Sigmoid()
mod['sigmoid']:forward(torch.randn(7, 15))

mod['lookup'] = nn.LookupTable(20, 100)
mod['lookup']:forward((torch.rand(3):abs()*20):int())

mod['reshape'] = nn.Reshape(4, 3, 7, true)
mod['reshape']:forward(torch.rand(7, 42, 2))

mod['splittable'] = nn.SplitTable(2, 2)
mod['splittable']:forward(torch.rand(7, 42, 2))

mod['replicate'] = nn.Replicate(3, 1, 1)
mod['replicate']:forward(torch.rand(7, 5))

torch.save("model1batch.t7", mod)