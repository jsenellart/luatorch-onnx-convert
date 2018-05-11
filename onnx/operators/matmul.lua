local MatMul, parent = torch.class('onnx.node.MatMul', 'onnx.node')

function MatMul:__init(inputs, outputs, precision)
  parent.__init(self, "MatMul", inputs, 2, outputs, 1)
  self._precision = precision
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function MatMul:getShapeConstraint(checker)
  checker:setChange(false)

  local ca = checker:assert2D(self._inputs[1])
  local cb = checker:assert2D(self._inputs[2])
  local cy = checker:assert2D(self._outputs[1])

  self._pass = checker:dimCheck(ca, 1, cy, 1) or checker:fail()
  self._pass = checker:dimCheck(ca, 2, cb, 1) or checker:fail()
  self._pass = checker:dimCheck(cb, 2, cy, 2) or checker:fail()

  return checker:hasChange()
end
