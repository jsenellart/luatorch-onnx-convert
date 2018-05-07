require 'nngraph'

local h1 = nn.Linear(20, 20, false)()
local h2 = nn.Linear(10, 10)()
local hh1 = nn.Linear(20, 1)(nn.Tanh()(h1))
local hh2 = nn.Linear(10, 1)(nn.Tanh()(h2))
local madd = nn.CAddTable()({hh1, hh2})
local oA = nn.Sigmoid()(madd)
local oB = nn.Tanh()(madd)
local gmod = nn.gModule({h1, h2}, {oA, oB})

torch.save("model3.t7", gmod)