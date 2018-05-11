local onnx_nn = require 'convertors.onnx_nngraph'
local convertor = require 'convertors.init'

function onnx_nn.Linear(obj, nInputs, nonbatch_mode)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Linear can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  if obj.bias == nil then
    local perms = { 0, 2, 1 }
    if nonbatch_mode or (obj and #obj.output:size()==1) then
      perms = { 1, 0 }
    end
    graph:add_node(onnx.node.Transpose.new({'b'}, {'bt'},
                                           perms))
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

function onnx_nn.MM(obj, nInputs, nonbatch_mode)
  local inputs = { 'a', 'b' }
  local graph = onnx.graph.new(inputs, {'y'})

  if obj.transA then
    inputs[1] = 'at'
    local perms = { 0, 2, 1 }
    if nonbatch_mode or (obj and #obj.output:size()==2) then
      perms = { 1, 0 }
    end
    graph:add_node(onnx.node.Transpose.new({'a'}, {'at'},
                                           perms))
  end
  if obj.transB then
    inputs[2] = 'bt'
    local perms = { 0, 2, 1 }
    if nonbatch_mode or (obj and #obj.output:size()==2)  then
      perms = { 1, 0 }
    end
    graph:add_node(onnx.node.Transpose.new({'b'}, {'bt'},
                                           perms))
  end
  graph:add_node(onnx.node.MatMul.new(inputs, {'y'}))
  return graph
end

function onnx_nn.Squeeze(obj, nInputs, nonbatch_mode)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Squeeze can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  local batch_offset = 1
  if nonbatch_mode or obj.numInputDims == nil then
    batch_offset = 0
  end
  graph:add_node(onnx.node.Squeeze.new({'x'}, {'y'}, { obj.dim - 1 + batch_offset }))
  return graph
end

function onnx_nn.SoftMax(obj, nInputs)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.SoftMax can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.SoftMax.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Identity(obj, nInputs)
  nInputs = nInputs or 1
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Identity.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Reshape(obj, nInputs, nonbatch_mode)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.Reshape can not have multiple inputs")
  local batchMode = obj.batchMode ~= false and not nonbatch_mode
  local reshape = obj.size:totable()
  if batchMode then
    table.insert(reshape, 1, 0)
  end
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Reshape.new({'x', 'ind'}, {'y'}, reshape))
  graph:add_initializer('ind', torch.Tensor(reshape))
  return graph
end

function onnx_nn.Replicate(obj, nInputs, nonbatch_mode)
  nInputs = nInputs or 1
  -- Unsqueeze and Tile
  assert(nInputs == 1, "nn.Replicate can not have multiple inputs")
  local graph = onnx.graph.new({'x'}, {'y'})
  local batch_offset = 1
  if nonbatch_mode or obj.ndim == nil then
    batch_offset = 0
  end
  graph:add_node(onnx.node.Unsqueeze.new({'x'}, {'xu'}, {obj.dim-1+batch_offset}))
  local repeats = torch.Tensor(obj.output:nDimension()):fill(1)
  repeats[obj.dim+batch_offset] = obj.nfeatures
  graph:add_node(onnx.node.Tile.new({'xu', 'repeats'}, {'y'}, repeats:totable()))
  graph:add_initializer('repeats', repeats)
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
  graph:add_node(onnx.node.Gather.new({'ind', 'x'}, {'y'},
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

function onnx_nn.MapTable(obj, nInputs, nonbatch_mode)
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
    local subgraph = convert_func(obj.modules[1], 1, nonbatch_mode)
    assert(#subgraph._outputs == 1)
    graph:merge(subgraph, i)
    graph:substitute_param(subgraph._inputs[1], inputs[i])
    graph:substitute_param(subgraph._outputs[1], outputs[i])
  end
  return graph
end

function onnx_nn.ConcatTable(obj, nInputs, nonbatch_mode)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.ConcatTable can not have multiple inputs")
  local inputs = {'x'}
  for i = 1, #obj.output do
    table.insert(outputs, 'y'..i)
  end
  local graph = onnx.graph.new(inputs, outputs)
  for i, subobj in pair(obj.modules) do
    local tname = convertor.mtype(subobj)
    local convert_func = convertor.isSupported(tname)
    if convert_func == nil then
      error('module `'..tname..'` not supported')
    end
    local subgraph = convert_func(subobj, 1, nonbatch_mode)
    assert(#subgraph._outputs == 1)
    graph:merge(subgraph, i)
    graph:substitute_param(subgraph._inputs[1], inputs[1])
    graph:substitute_param(subgraph._outputs[1], outputs[i])
  end
  return graph
end


function onnx_nn.JoinTable(obj, nInputs, nonbatch_mode)
  assert(nInputs ~= nil, "JoinTable can only be converted part of a gModule")
  local inputs = {}
  for i = 1, nInputs do
    table.insert(inputs, 'x'..i)
  end
  local batch_offset = 1
  if nonbatch_mode or obj.numInputDims == nil then
    batch_offset = 0
  end
  local graph = onnx.graph.new(inputs, {'y'})
  graph:add_node(onnx.node.Concat(inputs, {'y'}, obj.dimension-1+batch_offset))
  return graph
end

function onnx_nn.SplitTable(obj, nInputs, nonbatch_mode)
  nInputs = nInputs or 1
  assert(nInputs == 1, "nn.SplitTable can not have multiple inputs")
  local soutputs = {}  
  local outputs = {}
  assert(obj.output ~= nil, "can only convert model with outputs")
  for i = 1, #obj.output do
    table.insert(soutputs, 'sy'..i)    
    table.insert(outputs, 'y'..i)
  end
  local batch_offset = 1
  if nonbatch_mode or obj.numInputDims == nil then
    batch_offset = 0
  end
  local graph = onnx.graph.new({'x'}, outputs)
  graph:add_node(onnx.node.Split({'x'}, soutputs, obj.dimension-1+batch_offset))
  for i = 1, #obj.output do
    graph:add_node(onnx.node.Squeeze({'sy'..i}, {'y'..i}, {obj.dimension-1+batch_offset}))
  end
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
  local intMul = 'x1'
  for i = 2, nInputs do
    local resMul = 'y'
    if i < nInputs then
      resMul = 'y' .. i
    end
    graph:add_node(onnx.node.Mul.new({intMul, inputs[i]}, {resMul},
                                           onnx.helper.convertPrecision(obj.weight)))
    intMul = resMul
  end
  return graph
end

function onnx_nn.Sequential(obj, nInputs, nonbatch_mode)
  local subgraphs = {}
  for i = 1, #obj.modules do
    local obj = obj.modules[i]
    local tname = convertor.mtype(obj)
    if type(obj) == 'userdata' or type(obj) == 'table' then
      local convert_func = convertor.isSupported(tname)
      if convert_func then
        local subgraph = convert_func(obj, nInputs, nonbatch_mode)
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