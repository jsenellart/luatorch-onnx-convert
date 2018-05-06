local onnx_nn = {}

local convertor = require 'convertors.init'
require 'onnx.init'

function onnx_nn.gModule(obj)
  local graph = onnx.graph.new()

  local subgraphs = {}
  local nodeidx = {}

  for i, node in ipairs(obj.forwardnodes) do
    nodeidx[torch.pointer(node)] = i
    local object = node.data.module
    if object == nil then
      object = nn.Identity()
    end
    local tname = convertor.mtype(object)

    if type(object) == 'userdata' or type(object) == 'table' then
      local convert_func = convertor.isSupported(tname)
      if convert_func then
        local subgraph = convert_func(object)
        graph:merge(subgraph, i)
        subgraphs[torch.pointer(node)] = subgraph
      else
        error('operator '..tname..' not supported')
      end
    end
  end
  -- connecting output to inputs
  for i, node in ipairs(obj.forwardnodes) do
    for j, child in ipairs(node.children) do
      for _, p in ipairs(subgraphs[torch.pointer(node)]._outputs) do
        local pchild = subgraphs[torch.pointer(child)]._inputs[child.data.mapindex[node.data]]
        graph:substitute_param('n'..nodeidx[torch.pointer(child)]..'.'..pchild, 'n'..i..'.'..p)
      end
    end
  end

  return graph

end

return onnx_nn