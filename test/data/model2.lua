require 'nngraph'

local h1 = - nn.Linear(20,10)
local h2 = h1
     - nn.Tanh()
     - nn.Linear(10,10)
     - nn.Tanh()
     - nn.Linear(10, 1)
local mlp = nn.gModule({h1}, {h2})

torch.save("model2.t7", mlp)