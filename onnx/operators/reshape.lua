local Reshape, parent = torch.class('onnx.node.Reshape', 'onnx.node')

function Reshape:__init(inputs, outputs, fixedShape)
  parent.__init(self, "Reshape", inputs, 2, outputs, 1)
  self._fixedShape = fixedShape
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints

function Reshape:getShapeConstraint(checker)
  checker:setChange(false)

  local cx = checker:getParam(self._inputs[1])
  local cind = checker:assert1D(self._inputs[2])
  local cy = checker:getParam(self._outputs[1])

  -- reshape is not inversible - we can not infer shape of input given output
  if self._fixedShape ~= nil then
    if #cy == 0 then
      for i = 1, #self._fixedShape do
        if self._fixedShape[i] == -1 then
          table.insert(cy, checker:getUnkDimIdx())
        else
          table.insert(cy, self._fixedShape[i])
        end
      end
    else
      assert(#cy == #self._fixedShape, "invalid output shape")
      for i = 1, #self._fixedShape do
        if self._fixedShape[i] ~= -1 then
          checker:dimCheck(self._fixedShape, i, cy, i)
        end
      end
    end
  end

  return checker:hasChange()
end