local onnx_nn = require 'convertors.onnx_nngraph'

function onnx_nn.Linear(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Linear can not have multiple inputs")
  if obj.bias == nil then
    local graph = onnx.graph.new({'x'}, {'y'})
    graph:add_node(onnx.node.Transpose.new({'b'}, {'bt'},
                                           { 1, 0 }))
    graph:add_node(onnx.node.MatMul.new({'x', 'bt'}, {'y'},
                                        onnx.helper.convertPrecision(obj.weight)))
    graph:add_initializer('b', obj.weight)
    return graph
  else
    local graph = onnx.graph.new({'x'}, {'y'})
    graph:add_node(onnx.node.Gemm.new({'x', 'b', 'c'}, {'y'},
                                      onnx.helper.convertPrecision(obj.weight),
                                      1.0, -- alpha
                                      1.0, -- beta
                                      1,   -- broadcast C
                                      0,   -- transpose A
                                      1))  -- transpose B
    graph:add_initializer('b', obj.weight)
    graph:add_initializer('c', obj.bias)
    return graph
  end
end

function onnx_nn.Identity(_, nInputs)
  nInputs = nInputs or 1
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Identity.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Tanh(_, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Tanh can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Tanh.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Sigmoid(_, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Sigmoid can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Sigmoid.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.CAddTable(obj, nInputs)
  nInputs = nInputs or 2
  if nInputs == 1 then
    return onnx_nn.Identity(obj, 1)
  end
  local inputs = {}
  for i = 1, nInputs do
    table.insert(inputs, 'x'..i)
  end
  local graph = onnx.graph.new(inputs, {'y'})
  local intSum = 'x1'
  for i = 2, nInputs do
    local resSum = 'y'
    if i < nInputs then
      resSum = 'y' .. i
    end
    graph:add_node(onnx.node.Add.new({intSum, inputs[i]}, {resSum},
                                           onnx.helper.convertPrecision(obj.weight)))
    intSum = resSum
  end
  return graph
end

return onnx_nn