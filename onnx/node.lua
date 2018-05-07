local Node = torch.class('onnx.node')

function Node:__init(name, inputs, ninputs, outputs, noutputs)
  assert(#inputs==ninputs, "invalid number of inputs parameters")
  self._inputs = inputs
  assert(#outputs==noutputs, "invalid number of outputs parameters")
  self._outputs = outputs
  self._name = name
end

function Node:inputs()
  return self._inputs
end

function Node:outputs()
  return self._outputs
end

function Node:getShapeConstraint(_)
  error("getShapeConstraint not implemented - for operator "..self._name)
end

function Node:build(onnx_pb, node)
  for _, p in ipairs(self._inputs) do
    node.input:append(p)
  end
  for _, p in ipairs(self._outputs) do
    node.output:append(p)
  end
  node.op_type = self._name
end

function Node.addAttribute(onnx_node, name, namev, v, precision)
  local attribute = onnx_node.attribute:add()
  attribute.name = name
  attribute.type = precision
  if type(v) == "table" then
    for _, av in ipairs(v) do
      attribute[namev]:append(av)
    end
  else
    attribute[namev] = v
  end
end
