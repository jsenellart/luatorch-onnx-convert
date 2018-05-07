local Graph = torch.class('onnx.graph')
local paths = require 'paths'

function Graph:__init(inputs, outputs)
  self._nodes = {}
  self._node_input_map = {}
  self._node_output_map = {}
  self._node_map = {}
  self._initializer = {}
  self._inputs = inputs or {}
  self._outputs = outputs or {}
  self._checker = onnx.checker.new()
  self._tmpfile = paths.tmpname()

end

function Graph:add_node(node)
  table.insert(self._nodes, node)
  self._node_map[torch.pointer(node)] = node
  for _, p in ipairs(node:inputs()) do
    if self._node_input_map[p] == nil then
      self._node_input_map[p] = {}
    end
    table.insert(self._node_input_map[p], torch.pointer(node))
  end
  for _, p in ipairs(node:outputs()) do
    self._node_output_map[p] = torch.pointer(node)
  end
end

function Graph:add_initializer(p, obj)
  assert(self._node_input_map[p] ~= nil or self._node_output_map[p] ~= nil, "unknown param `"..p.."`")
  assert(self._initializer[p] == nil, "two initializers defined for param `"..p.."`")
  self._initializer[p] = obj
  self._checker:setDims(p, torch.totable(obj:size()))
end

function Graph:substitute_param(p1, p2)
  for _, n in ipairs(self._nodes) do
    for i,v in ipairs(n._inputs) do
      if v == p1 then
        n._inputs[i] = p2
      end
    end
    for i,v in ipairs(n._outputs) do
      if v == p1 then
        n._outputs[i] = p2
      end
    end
  end
  for i,v in ipairs(self._inputs) do
    if v == p1 then
      self._inputs[i] = p2
    end
  end
  for i,v in ipairs(self._outputs) do
    if v == p1 then
      self._outputs[i] = p2
    end
  end
end

function Graph:merge(subgraph, idx)
  for i, v in ipairs(subgraph._inputs) do
    subgraph._inputs[i] = 'n'..idx..'.'..v
  end
  for i, v in ipairs(subgraph._outputs) do
    subgraph._outputs[i] = 'n'..idx..'.'..v
  end
  for _, n in ipairs(subgraph._nodes) do
    table.insert(self._nodes, n)
    self._node_map[torch.pointer(n)] = n
    for i,v in ipairs(n._inputs) do
      n._inputs[i] = 'n'..idx..'.'..v
    end
    for i,v in ipairs(n._outputs) do
      n._outputs[i] = 'n'..idx..'.'..v
    end
  end
  for param, pn in pairs(subgraph._node_input_map) do
    if self._node_input_map['n'..idx..'.'..param] == nil then
      self._node_input_map['n'..idx..'.'..param] = {}
    end
    table.insert(self._node_input_map['n'..idx..'.'..param], pn)
  end
  for param, pn in pairs(subgraph._node_output_map) do
    if self._node_output_map['n'..idx..'.'..param] == nil then
      self._node_output_map['n'..idx..'.'..param] = {}
    end
    table.insert(self._node_output_map['n'..idx..'.'..param], pn)
  end
  for param, obj in pairs(subgraph._initializer) do
    self:add_initializer('n'..idx..'.'..param, obj)
  end
end

function Graph:build(onnx_pb, onnx_graph)

  -- shape inference
  local change = true
  while change do
    change = false
    for _, n in pairs(self._nodes) do
      change = n:getShapeConstraint(self._checker) or change
    end
  end

  -- build the graph - input params
  for _, p in ipairs(self._inputs) do
    local input = onnx_graph.input:add()
    input.name = p
    input.type.tensor_type.elem_type = onnx_pb.TensorProto.FLOAT
    -- needed because of bug in protobuf library
    input.type:_Modified(true)
    for _, d in ipairs(self._checker:params()[p]) do
      input.type.tensor_type.shape.dim:add().dim_value = d
    end
  end

  -- add parameters for which we have initializer
  for p, _ in pairs(self._initializer) do
    local input = onnx_graph.input:add()
    input.name = p
    input.type.tensor_type.elem_type = onnx_pb.TensorProto.FLOAT
    -- needed because of bug in protobuf library
    input.type:_Modified(true)
    for _, d in ipairs(self._checker:params()[p]) do
      input.type.tensor_type.shape.dim:add().dim_value = d
    end
  end

  -- build the graph - output params
  for _, p in ipairs(self._outputs) do
    local output = onnx_graph.output:add()
    output.name = p
    output.type.tensor_type.elem_type = onnx_pb.TensorProto.FLOAT
    -- needed because of bug in protobuf library
    output.type:_Modified(true)
    for _, d in ipairs(self._checker:params()[p]) do
      output.type.tensor_type.shape.dim:add().dim_value = d
    end
  end

  -- build the graph - the actual nodes
  for _, n in pairs(self._nodes) do
    local node = onnx_graph.node:add()
    n:build(onnx_pb, node)
  end

  -- dump initializer
  for p, w in pairs(self._initializer) do
    w = w:float()
    local initializer = onnx_graph.initializer:add()
    for _, d in ipairs(self._checker:params()[p]) do
      initializer.dims:append(d)
    end
    initializer.data_type = onnx_pb.TensorProto.FLOAT
    initializer.name = p
    local file = torch.DiskFile(self._tmpfile, 'w'):binary()
    file:writeFloat(w:storage().new(w:storage(), w:storageOffset(), w:nElement()))
    file:close()
    local inp = assert(io.open(self._tmpfile, "rb"))
    initializer.raw_data = inp:read("*all")
  end

end