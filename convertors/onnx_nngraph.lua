local onnx_nn = {}

local convertor = require 'convertors.init'
require 'onnx.init'

function serialize(t)
  if type(t) == 'table' then
    local s = ''
    for i,v in pairs(t) do
      if s ~= '' then
        s = s..','
      else
        s = ''
      end
      s =s .. i..':'..serialize(v)
    end
    return '{'..s..'}'
  else
    return t
  end
end

function onnx_nn.gModule(obj, _, nonbatch_mode)
  local inputs = {}
  local outputs = {}
  for i = 1, obj.nInputs do
    table.insert(inputs, "x" .. i)
  end

  for i = 1, #obj.outnode.children do
    table.insert(outputs, "y" .. i)
  end

  local graph = onnx.graph.new(inputs, outputs)

  local function neteval(idx, node)
    local function propagate(node, x)
      for i, child in ipairs(node.children) do
        child.data.input = child.data.input or {}
        local mapindex = child.data.mapindex[node.data]
        assert(not child.data.input[mapindex], "each input should have one source")
        child.data.input[mapindex] = x
      end
    end
    if node.data.selectindex then
      assert(not node.data.module, "the selectindex-handling nodes should have no module")
      local input = node.data.input
      input = input[1][node.data.selectindex]
      propagate(node, input)
    else
      local inputs = node.data.input
      -- a parameter node is captured
      if inputs == nil and node.data.module ~= nil then
        inputs = {}
      end
      if #inputs == 1 then
        inputs = inputs[1]
      end

      -- forward through this node
      -- If no module is present, the node behaves like nn.Identity.
      local outputs
      if not node.data.module then
        outputs = inputs
      else
        local object = node.data.module
        local tname = convertor.mtype(object)
        if type(object) == 'userdata' or type(object) == 'table' then
          local convert_func = convertor.isSupported(tname)
          if convert_func then
            if type(inputs) ~= 'table' then
              inputs = { inputs }
            end
            local subgraph = convert_func(object, #inputs, nonbatch_mode)
            outputs = subgraph._outputs
            graph:merge(subgraph, idx)
            for i = 1, #inputs do
              graph:substitute_param(subgraph._inputs[i], inputs[i])
            end
          else
            error('module `'..tname..'` not supported')
          end
        end
      end
      if #outputs == 1 then
        outputs = outputs[1]
      end
      propagate(node, outputs)
    end
  end

  local innode = obj.innode

  -- first clear the input states
  for _, node in ipairs(obj.forwardnodes) do
    local input = node.data.input
    while input and #input>0 do
       table.remove(input)
    end
  end
  -- Set the starting input.
  -- We do copy instead of modifying the passed input.
  obj.innode.data.input = obj.innode.data.input or {}
  for i, item in ipairs(inputs) do
    obj.innode.data.input[i] = item
  end

  -- the run forward
  for i, node in ipairs(obj.forwardnodes) do
    neteval(i, node)
  end

  for i, p in ipairs(outputs) do
    graph:substitute_param(obj.outnode.data.input[i], p)
  end

  return graph

end

return onnx_nn