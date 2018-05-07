local Transpose, parent = torch.class('onnx.node.Transpose', 'onnx.node')

function Transpose:__init(inputs, outputs, perm)
  parent.__init(self, "Transpose", inputs, 1, outputs, 1)
  self._perm = perm
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Transpose:getShapeConstraint(checker)
  local cx = checker:assert2D(self._inputs[1])
  local cy = checker:assert2D(self._outputs[1])

  local count = 0
  checker:setChange(true)
  while checker:hasChange() do
    count = count + 1
    checker:setChange(false)
    self._pass = checker:dimCheck(cx, 1, cy, 2) or checker:fail()
    self._pass = checker:dimCheck(cx, 2, cy, 1) or checker:fail()
  end

  return count ~= 1
end

function Transpose:build(onnx_pb, node)
  parent.build(self, node)
  self.addAttribute(node, "perm", 'ints', self._perm, onnx_pb.AttributeProto.INTS)
end