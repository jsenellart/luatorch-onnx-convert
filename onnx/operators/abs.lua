local Abs, parent = torch.class('onnx.node.Abs', 'onnx.node')

function Abs:__init(inputs, outputs)
  parent.__init(self, "Abs", inputs, 1, outputs, 1)
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Abs:getShapeConstraint(checker)
  local cx = checker:getParam(self._inputs[1])
  local cy = checker:getParam(self._outputs[1])

  checker:setChange(false)
  self._pass = checker:sameShape({cx, cy}) or checker:fail()

  return checker:hasChange()
end