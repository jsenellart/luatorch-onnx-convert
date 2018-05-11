require 'nngraph'

local id1 = nn.Identity()()
local id2 = nn.Identity()()

local a12 = nn.CAddTable()({id1, id2})
local mod1 = nn.gModule({id1,id2}, {a12, nn.Sigmoid()(id1)})

local id3 = nn.Identity()()
local id4 = nn.Identity()()
local m12 = nn.CMulTable()({id3, id4})
local o1, o2 = mod1({m12, id3}):split(2)
local mod2 = nn.gModule({id3, id4}, {m12, o1, o2})

mod2:forward({torch.randn(5), torch.randn(5)})

torch.save("model5.t7", mod2)