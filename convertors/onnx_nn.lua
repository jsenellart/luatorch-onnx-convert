local onnx_nn = require 'convertors.onnx_nngraph'
local convertor = require 'convertors.init'

function onnx_nn.Linear(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Linear can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  if obj.bias == nil then
    graph:add_node(onnx.node.Transpose.new({'b'}, {'bt'},
                                           { 1, 0 }))
    graph:add_node(onnx.node.MatMul.new({'x', 'bt'}, {'y'},
                                        onnx.helper.convertPrecision(obj.weight)))
  else
    graph:add_node(onnx.node.Gemm.new({'x', 'b', 'c'}, {'y'},
                                      onnx.helper.convertPrecision(obj.weight),
                                      1.0, -- alpha
                                      1.0, -- beta
                                      1,   -- broadcast C
                                      0,   -- transpose A
                                      1))  -- transpose B
    graph:add_initializer('c', obj.bias)
  end
  graph:add_initializer('b', obj.weight)
  return graph
end

function onnx_nn.MM(obj, nInputs)
  local inputs = { 'a', 'b' }
  local graph = onnx.graph.new(inputs, {'y'})

  if obj.transA then
    inputs[1] = 'at'
    graph:add_node(onnx.node.Transpose.new({'a'}, {'at'},
                                           { 1, 0 }))
  end
  if obj.transB then
    inputs[2] = 'bt'
    graph:add_node(onnx.node.Transpose.new({'b'}, {'bt'},
                                           { 1, 0 }))
  end
  graph:add_node(onnx.node.MatMul.new(inputs, {'y'}))
  return graph
end

function onnx_nn.Squeeze(obj, nInputs)
  local inputs = { 'a', 'b' }
  local graph = onnx.graph.new(inputs, {'y'})

  graph:add_node(onnx.node.Squeeze.new({'x'}, {'y'}, { obj.dim }))

  return graph
end

function onnx_nn.SoftMax(obj, nInputs)
  local inputs = { 'a', 'b' }
  local graph = onnx.graph.new(inputs, {'y'})
  graph:add_node(onnx.node.SoftMax.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Identity(obj, nInputs)
  nInputs = nInputs or 1
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Identity.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Reshape(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Reshape can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Reshape.new({'x', 'ind'}, {'y'}, obj.size:totable()))
  graph:add_initializer('ind', torch.Tensor(obj.size:totable()))
  return graph
end

function onnx_nn.Replicate(obj, nInputs)
  nInputs = nInputs or 1
  -- Reshape and Tile
  assert(nInputs == 1, "nn.Reshape can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  -- local repeats = torch.Tensor(obj.output:nDimension()-1):fill(1)
  -- repeats[obj.dim] = obj.nfeatures
  -- graph:add_node(onnx.node.Tile.new({'x', 'repeats'}, {'y'}, repeats:totable()))
  -- graph:add_initializer('repeats', repeats)
  -- graph:set_dimension('y', obj.size:totable())
  return graph
end

function onnx_nn.Abs(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Abs can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Abs.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.LookupTable(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Lookup can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Gather.new({'x', 'ind'}, {'y'},
                                   onnx.helper.convertPrecision(obj.weight),
                                   0)) -- axis
  graph:add_initializer('ind', obj.weight)
  graph._checker:assert1D('x')
  return graph
end

function onnx_nn.Tanh(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Tanh can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Tanh.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Sigmoid(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Sigmoid can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Sigmoid.new({'x'}, {'y'}))
  return graph
end

-- convert Dropout to identity
function onnx_nn.Dropout(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Dropout can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Identity.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.MapTable(obj, nInputs)
  if nInputs == nil then
    nInputs = #obj.output
  end
  local tname = convertor.mtype(obj.modules[1])
  local convert_func = convertor.isSupported(tname)
  if convert_func == nil then
    error('module `'..tname..'` not supported')
  end
  local inputs = {}
  local outputs = {}
  for i = 1, nInputs do
    table.insert(inputs, "x"..i)
    table.insert(outputs, "y"..i)
  end
  local graph = onnx.graph.new(inputs, outputs)
  for i = 1, nInputs do
    local subgraph = convert_func(obj.modules[1], 1)
    assert(#subgraph._outputs == 1)
    graph:merge(subgraph, i)
    graph:substitute_param(subgraph._inputs[1], inputs[i])
    graph:substitute_param(subgraph._outputs[1], outputs[i])
  end
  return graph
end

function onnx_nn.JoinTable(obj, nInputs)
  assert(nInputs ~= nil, "JoinTable can only be converted part of a gModule")
  local inputs = {}
  for i = 1, nInputs do
    table.insert(inputs, 'x'..i)
  end
  local graph = onnx.graph.new(inputs, {'y'})
  graph:add_node(onnx.node.Concat(inputs, {'y'}, obj.dimension-1))
  return graph
end

function onnx_nn.SplitTable(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.SplitTable can not have multiple inputs")
  local soutputs = {}  
  local outputs = {}
  assert(obj.output ~= nil, "can only convert model with outputs")
  for i = 1, #obj.output do
    table.insert(soutputs, 'sy'..i)    
    table.insert(outputs, 'y'..i)
  end
  local graph = onnx.graph.new({'x'}, outputs)
  graph:add_node(onnx.node.Split({'x'}, soutputs, obj.dimension-1))
  local outputDim = obj.output[1]:size():totable()
  local outputSplit = obj.output[1]:size():totable()
  table.insert(outputSplit, obj.dimension, 1)
  for i = 1, #obj.output do
    graph:add_node(onnx.node.Reshape({'sy'..i, 'ind'}, {'y'..i}))
    graph:set_dimension('y'..i, outputSplit)
  end
  graph:add_initializer('ind', torch.Tensor(outputDim))
  return graph
end

function onnx_nn.ConcatTable(obj, nInputs)
  local graph = onnx.graph.new()
  local inputs = {}
  for i = 1, nInputs do
    table.insert(inputs, 'x'..i)
  end
  graph:add_node(onnx.node.Concat.new(inputs, {'y'}))
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

function onnx_nn.CMulTable(obj, nInputs)
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
    graph:add_node(onnx.node.Mul.new({intSum, inputs[i]}, {resSum},
                                           onnx.helper.convertPrecision(obj.weight)))
    intSum = resSum
  end
  return graph
end

function onnx_nn.Sequential(obj, nInputs)
  local subgraphs = {}
  for i = 1, #obj.modules do
    local obj = obj.modules[i]
    local tname = convertor.mtype(obj)
    if type(obj) == 'userdata' or type(obj) == 'table' then
      local convert_func = convertor.isSupported(tname)
      if convert_func then
        local subgraph = convert_func(obj, nInputs)
        nInputs = #subgraph._outputs
        table.insert(subgraphs, subgraph)
      else
        error('module `'..tname..'` not supported')
      end
    else
      error("unsupported module in nn.Sequential: `"+tname+"`")
    end
  end
  local inputs = {}
  local outputs = {}
  for i = 1, #subgraphs[1]._inputs do
    table.insert(inputs, "x"..i)
  end
  for i = 1, #subgraphs[#subgraphs]._outputs do
    table.insert(outputs, "y"..i)
  end
  local graph = onnx.graph.new(inputs, outputs)
  for i, subgraph in ipairs(subgraphs) do
    graph:merge(subgraph, i)
    for i = 1, #inputs do
      graph:substitute_param(subgraph._inputs[i], inputs[i])
    end
    inputs = subgraph._outputs
  end
  for i = 1, #outputs do
    graph:substitute_param(subgraphs[#subgraphs]._outputs[i], outputs[i])
  end
  return graph
end

return onnx_nn