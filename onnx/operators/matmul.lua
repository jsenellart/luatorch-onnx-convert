local MatMul, parent = torch.class('onnx.node.MatMul', 'onnx.node')

function MatMul:__init(inputs, outputs, precision)
  parent.__init(self, "MatMul", inputs, 2, outputs, 1)
  self._precision = precision
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function MatMul:getShapeConstraint(checker)
  local ca = checker:assert2D(self._inputs[1])
  local cb = checker:assert2D(self._inputs[2])
  local cy = checker:assert2D(self._outputs[1])

  local count = 0
  checker:setChange(true)
  while checker:hasChange() do
    count = count + 1
    checker:setChange(false)
    _ = checker:dimCheck(ca, 1, cy, 1) or checker:fail()
    _ = checker:dimCheck(ca, 2, cb, 1) or checker:fail()
    _ = checker:dimCheck(cb, 2, cy, 2) or checker:fail()
  end

  return count ~= 1
end

function MatMul:build(node)
  parent.build(self, node)
end