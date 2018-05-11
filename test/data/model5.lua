require 'nngraph'

local id1 = nn.Identity()()
local id2 = nn.Identity()()

local jt = nn.JoinTable(2, 2)({id1, id2})
local mod = nn.gModule({id1,id2}, {jt})

mod:forward({torch.rand(2,10), torch.rand(2,20)})

torch.save("model5.t7", mod)