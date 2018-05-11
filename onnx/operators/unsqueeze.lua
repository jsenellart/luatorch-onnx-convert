local Unsqueeze, parent = torch.class('onnx.node.Unsqueeze', 'onnx.node')

function Unsqueeze:__init(inputs, outputs, axes)
  parent.__init(self, "Unsqueeze", inputs, 1, outputs, 1)
  self._axes = axes
end

function _find(v, t)
  for _, x in ipairs(t) do
    if x == v then
      return true
    end
  end
  return false
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Unsqueeze:getShapeConstraint(checker)
  checker:setChange(false)

  local cx = checker:getParam(self._inputs[1])
  local cy = checker:getParam(self._outputs[1])

  if #cy == 0 and #cx ~= 0 then
    for i = 1, #cx do
      table.insert(cy, cx[i])
    end
    for i = 1, #self._axes do
      table.insert(cy, i+1, 1)
    end
    checker:setChange(true)
  elseif #cy ~= 0 and #cx == 0 then
    for i = 1, #cy do
      if not _find(i-1, self._axes) then
        table.insert(cx, cy[i])
      end
    end
    checker:setChange(true)
  elseif #cx ~= 0 and #cy ~= 0 then
    assert(#cx==#cy-#self._axes, "invalid shapes")
  end

  return checker:hasChange()
end

function Unsqueeze:build(onnx_pb, node)
  parent.build(self, onnx_pb, node)
  self.addAttribute(node, "axes", 'ints', self._axes, onnx_pb.AttributeProto.INTS)
end