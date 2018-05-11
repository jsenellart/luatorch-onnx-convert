local Gather, parent = torch.class('onnx.node.Gather', 'onnx.node')

function Gather:__init(inputs, outputs, precision, axis)
  parent.__init(self, "Gather", inputs, 2, outputs, 1)
  self._precision = precision
  self._axis = axis
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Gather:getShapeConstraint(checker)
  local cind = checker:getParam(self._inputs[1])
  local cx = checker:getParam(self._inputs[2])
  local cy = checker:getParam(self._outputs[1])

  checker:setType(self._inputs[1], "FLOAT")
  checker:setType(self._inputs[2], "INT32")
  checker:setType(self._outputs[1], "FLOAT")

  checker:setChange(false)

  if #cx ~= 0 and #cind ~= 0 then
    if #cy == 0 then
      for i = 1, #cx do
        table.insert(cy, cx[i])
      end
      for i = 1, #cind - 1 do
        table.insert(cy, cind[i+1])
      end
      checker:setChange(true)
    else
      assert(#cy == #cx + #cind - 1, "invalid size of gather output")
      for i = 1, #cx do
        checker:dimCheck(cy, i, cx, i)
      end
      for i = 1, #cind - 1 do
        checker:dimCheck(cy, i+#cx-1, cind, i+1 )
      end
    end
  elseif #cy ~= 0 then
    checker:setChange(true)
    if #cx == 0 then
      for i = 1, #cy - #cind + 1 do
        table.insert(cx, checker:getUnkDimIdx())
      end
    else
      for i = 1, #cy - #cx + 1 do
        table.insert(cind, checker:getUnkDimIdx())
      end
    end
  end

  return checker:hasChange()
end

function Gather:build(onnx_pb, node)
  parent.build(self, onnx_pb, node)
  self.addAttribute(node, "axis", 'i', self._axis, onnx_pb.AttributeProto.INT)
end