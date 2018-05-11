local Split, parent = torch.class('onnx.node.Split', 'onnx.node')

function Split:__init(inputs, outputs, axis)
  parent.__init(self, "Split", inputs, 1, outputs, #outputs)
  self._axis = axis
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Split:getShapeConstraint(checker)
  checker:setChange(false)

  local cx = checker:getParam(self._inputs[1])

  local nbDimOutput
  local cys = {}
  for _, p in pairs(self._outputs) do
    local cy = checker:getParam(p)
    if #cy ~= 0 then
      assert(cy[self._axis+1] == 1, "inconsistent size of axis output")
      if nbDimOuput == nil then
        nbDimOutput = #cy
      else
        assert(nbDimOutput == #cy, "inconsistent dimension of Split output")
      end
    end
    table.insert(cys, cy)
  end

  if nbDimOutput then
    checker:sameShape(cys)
    if #cx == 0 then
      for i, d in pairs(cys[1]) do
        if i-1 == self._axis then
          table.insert(cx, #self._outputs)
        else
          table.insert(cx, d)
        end
      end
    else
      assert(#cx == #cys[1], "incorrect dimension of SplitTable input")
      for i, d in pairs(cys[1]) do
        if i-1 ~= self._axis then
          checker:dimCheck(cx, i, cys[1], i)
        else
          if cx[i] < 0 then
            checker:changeUnk(cx[i], #self._outputs)
          else
            assert(cx[i] == #self._outputs, "invalid split axis dimension")
          end
        end
      end
    end
  end

  return checker:hasChange()
end