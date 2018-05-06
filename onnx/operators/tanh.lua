local Tanh, parent = torch.class('onnx.node.Tanh', 'onnx.node')

function Tanh:__init(inputs, outputs, precision)
  parent.__init(self, "Tanh", inputs, 1, outputs, 1)
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Tanh:getShapeConstraint(checker)
  local cx = checker:getParam(self._inputs[1])
  local cy = checker:getParam(self._outputs[1])

  checker:setChange(false)
  _ = checker:sameShape({cx, cy}) or checker:fail()

  return checker:hasChange()
end