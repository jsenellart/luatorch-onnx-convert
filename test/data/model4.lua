require 'nn'

local mlp = nn.Sequential()
mlp:add(nn.Linear(10, 25)) -- Linear module (10 inputs, 25 hidden units)
mlp:add(nn.Tanh())         -- apply hyperbolic tangent transfer function on each hidden units
mlp:add(nn.Linear(25, 1))  -- Linear module (25 inputs, 1 output)

torch.save("model4.t7", mlp)