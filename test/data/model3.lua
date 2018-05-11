require 'nngraph'

function _buildLayer(inputSize, hiddenSize)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local prevC = inputs[1]
  local prevH = inputs[2]
  local x = inputs[3]

  -- Evaluate the input sums at once for efficiency.
  local i2h = nn.Linear(inputSize, 4 * hiddenSize)(x)
  local h2h = nn.Linear(hiddenSize, 4 * hiddenSize)(prevH)
  local allInputSums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, hiddenSize)(allInputSums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

  -- Decode the gates.
  local inGate = nn.Sigmoid()(n1)
  local forgetGate = nn.Sigmoid()(n2)
  local outGate = nn.Sigmoid()(n3)

  -- Decode the write inputs.
  local inTransform = nn.Tanh()(n4)

  -- Perform the LSTM update.
  local nextC = nn.CAddTable()({
    nn.CMulTable()({forgetGate, prevC}),
    nn.CMulTable()({inGate, inTransform})
  })

  -- Gated cells form the output.
  local nextH = nn.CMulTable()({outGate, nn.Tanh()(nextC)})

  return nn.gModule(inputs, {nextC, nextH})
end

local gmod = _buildLayer(10,5)

gmod:forward({torch.randn(3, 5), torch.randn(3,5), torch.randn(3,10)})

torch.save("model3.t7", gmod)