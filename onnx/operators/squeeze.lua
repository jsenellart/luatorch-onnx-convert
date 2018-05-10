local Squeeze, parent = torch.class('onnx.node.Squeeze', 'onnx.node')

function Squeeze:__init(inputs, outputs, axis)
  parent.__init(self, "Squeeze", inputs, 1, outputs, 1)
  self._axis = axis
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Squeeze:getShapeConstraint(checker)
  checker:setChange(false)

  local cx = checker:getParam(self._inputs[1])
  local cy = checker:getParam(self._outputs[1])

  if #cx == 0 and #cy ~=0 then
    for i = 1, #cy+#self._axis do
      table.insert(cx, checker:getUnkDimIdx())
    end
    checker:setChange(true)
  elseif #cx ~= 0 and #cy == 0 then
    for i = 1, #cx-#self._axis do
      table.insert(cx, checker:getUnkDimIdx())
    end
    checker:setChange(true)
  elseif #cx ~= 0 and #cy ~= 0 then
    assert(#cy==#cx-#self._axis, "invalid shapes")
  end

  return checker:hasChange()
end

function Squeeze:build(onnx_pb, node)
  parent.build(self, onnx_pb, node)
  self.addAttribute(node, "axis", 'ints', self._axis, onnx_pb.AttributeProto.INTS)
end