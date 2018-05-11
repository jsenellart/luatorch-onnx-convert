local Tile, parent = torch.class('onnx.node.Tile', 'onnx.node')

function Tile:__init(inputs, outputs, fixedRepeats)
  parent.__init(self, "Tile", inputs, 2, outputs, 1)
  self._fixedRepeats = fixedRepeats
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints

function Tile:getShapeConstraint(checker)
  checker:setChange(false)

  local cx = checker:getParam(self._inputs[1])
  local crepeats = checker:assert1D(self._inputs[2])
  local cy = checker:getParam(self._outputs[1])

  if self._fixedRepeats then
    local n = #self._fixedRepeats
    assert(crepeats[1] == n, 'pb with tile initialization')
    cx = checker:assertND(self._inputs[1], n) or checker:fail()
    cy = checker:assertND(self._outputs[1], n) or checker:fail()
    for i = 1, n do
      if cx[i] > 0 then
        if cy[i] > 0 then
          assert(cy[i] == cx[i] * self._fixedRepeats[i], 'invalid tile result')
        else
          checker:changeUnk(cy[i], cx[i] * self._fixedRepeats[i])
        end
      elseif cy[i] > 0 then
        assert(cy[i] % self._fixedRepeats[i] == 0, 'tile size not consistent with multiplier')
        checker:changeUnk(cx[i], cy[i] / self._fixedRepeats[i])
      end
    end
  end

  return checker:hasChange()
end