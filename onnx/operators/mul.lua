local Mul, parent = torch.class('onnx.node.Mul', 'onnx.node')

function Mul:__init(inputs, outputs, precision)
  parent.__init(self, "Mul", inputs, 2, outputs, 1)
  self._precision = precision
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Mul:getShapeConstraint(checker)
  checker:setChange(false)

  local cx1 = checker:getParam(self._inputs[1])
  local cx2 = checker:getParam(self._inputs[2])
  local cy = checker:getParam(self._outputs[1])

  self._pass = checker:sameShape({cx1, cx2, cy}) or checker:fail()

  return checker:hasChange()
end
