local onnx_nn = {}

local convertor = require 'convertors.init'
require 'onnx.init'

-- search recursively for refp parents, and select corresponding mapindex
local function _fix_label_rec(refp, p, mapIndex, selectindex, graph, parent, nodes, subgraphs)
  for _, q in pairs(parent[p]) do
    mapIndex =  nodes[p].data.mapindex[nodes[q].data] or mapIndex
    if parent[q] ~= nil and not nodes[q].data.module then
      return _fix_label_rec(refp, q, mapIndex, selectindex, graph, parent, nodes, subgraphs)
    end
    local qoutputs
    if parent[q] == nil then
      qoutputs = graph._inputs
    else
      qoutputs = subgraphs[q]._outputs
    end
    local ninputs
    if subgraphs[refp] then
      ninputs = subgraphs[refp]._inputs
    else
      ninputs = graph._outputs
    end
    if selectindex then
      graph:substitute_param(ninputs[mapIndex], qoutputs[selectindex])
    else
      graph:substitute_param(ninputs[mapIndex], qoutputs[1])
    end
  end
end

function onnx_nn.gModule(obj)
  local inputs = {}
  local outputs = {}
  for i = 1, obj.nInputs do
    table.insert(inputs, "x" .. i)
  end
  for i = 1, #obj.outnode.children do
    table.insert(outputs, "y" .. i)
  end

  local graph = onnx.graph.new(inputs, outputs)

  local subgraphs = {}
  local nodes = {}
  local nInput = {}
  local selectInput = {}
  local parent = {}

  -- define number of inputs on the first node
  nInput[torch.pointer(obj.forwardnodes[1])] = obj.nInputs

  for i, node in ipairs(obj.forwardnodes) do
    nodes[torch.pointer(node)] = node
    local object = node.data.module
    if node.data.selectindex ~= nil then
      for _, child in ipairs(node.children) do
        nInput[torch.pointer(child)] = 1
        selectInput[torch.pointer(child)] = node.data.selectindex
        if parent[torch.pointer(child)] == nil then
          parent[torch.pointer(child)] = {}
        end
        table.insert(parent[torch.pointer(child)], torch.pointer(node))
      end
    elseif object == nil then
      -- this node is just an identity (initial node most likely)
      -- we don't want to create a useless Identity - let us just
      -- forward all the input nodes
      for _, child in ipairs(node.children) do
        selectInput[torch.pointer(child)] = selectInput[torch.pointer(node)]
        nInput[torch.pointer(child)] = nInput[torch.pointer(node)]
        if parent[torch.pointer(child)] == nil then
          parent[torch.pointer(child)] = {}
        end
        table.insert(parent[torch.pointer(child)], torch.pointer(node))
      end
      if #node.children == 0 then
        if parent[torch.pointer(obj.outnode)] == nil then
          parent[torch.pointer(obj.outnode)] = {}
        end
        table.insert(parent[torch.pointer(obj.outnode)], torch.pointer(node))
      end
    else
      local tname = convertor.mtype(object)
      if type(object) == 'userdata' or type(object) == 'table' then
        local convert_func = convertor.isSupported(tname)
        if convert_func then
          local subgraph = convert_func(object, nInput[torch.pointer(node)])
          graph:merge(subgraph, i)
          subgraphs[torch.pointer(node)] = subgraph
          for _, child in ipairs(node.children) do
            if nInput[torch.pointer(child)] == nil then
              nInput[torch.pointer(child)] = 0
            end
            if #subgraph._outputs > nInput[torch.pointer(child)] then
              nInput[torch.pointer(child)] = #subgraph._outputs
            end
            if child.data.mapindex[node.data] > nInput[torch.pointer(child)] then
              nInput[torch.pointer(child)] = child.data.mapindex[node.data]
            end
            if parent[torch.pointer(child)] == nil then
              parent[torch.pointer(child)] = {}
            end
            table.insert(parent[torch.pointer(child)], torch.pointer(node))
          end
        else
          error('operator '..tname..' not supported')
        end
      end
    end
  end
  subgraphs[torch.pointer(obj.outnode)] = false
  nodes[torch.pointer(obj.outnode)] = obj.outnode
  -- connecting output to inputs
  for p, _ in pairs(subgraphs) do
    local selectindex = selectInput[p]
    _fix_label_rec(p, p, nil, selectindex, graph, parent, nodes, subgraphs)
  end

  return graph

end

return onnx_nn