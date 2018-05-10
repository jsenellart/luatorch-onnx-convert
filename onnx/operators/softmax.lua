local SoftMax, parent = torch.class('onnx.node.SoftMax', 'onnx.node')

function SoftMax:__init(inputs, outputs)
  parent.__init(self, "SoftMax", inputs, 1, outputs, 1)
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function SoftMax:getShapeConstraint(checker)
  checker:setChange(false)

  local cx = checker:getParam(self._inputs[1])
  local cy = checker:getParam(self._outputs[1])

  self._pass = checker:sameShape({cx, cy}) or checker:fail()

  return checker:hasChange()
end