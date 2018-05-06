local Add, parent = torch.class('onnx.node.Add', 'onnx.node')

function Add:__init(inputs, outputs, precision)
  parent.__init(self, "Add", inputs, 2, outputs, 1)
  self._precision = precision
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Add:getShapeConstraint(checker)
  local cx1 = checker:getParam(self._inputs[1])
  local cx2 = checker:getParam(self._inputs[2])
  local cy = checker:getParam(self._outputs[1])

  checker:setChange(false)
  _ = checker:sameShape({cx1, cx2, cy}) or checker:fail()

  return checker:hasChange()
end

function Add:build(node)
  parent.build(self, node)
end