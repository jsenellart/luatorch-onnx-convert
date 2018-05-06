local Identity, parent = torch.class('onnx.node.Identity', 'onnx.node')

function Identity:__init(inputs, outputs)
  parent.__init(self, "Identity", inputs, 1, outputs, 1)
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Identity:getShapeConstraint(checker)
  local cx = checker:getParam(self._inputs[1])
  local cy = checker:getParam(self._outputs[1])

  if #cx == 0 and cy ~=0 then
    for _, v in pairs(cy) do
      table.insert(cx, v)
    end
  end

  if #cy == 0 and cx ~=0 then
    for _, v in pairs(cx) do
      table.insert(cy, v)
    end
  end

  local count = 0
  checker:setChange(true)
  while checker:hasChange() do
    count = count + 1
    checker:setChange(false)
    for i = 1, #cx do
      _ = checker:dimCheck(cx, i, cy, i) or checker:fail()
    end
  end

  return count ~= 1
end