local Tile, parent = torch.class('onnx.node.Tile', 'onnx.node')

function Tile:__init(inputs, outputs, fixedRepeats)
  parent.__init(self, "Tile", inputs, 2, outputs, 1)
  self._fixedRepeats = fixedRepeats
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints

function Tile:getShapeConstraint(checker)
  checker:setChange(false)

  -- TODO

  return checker:hasChange()
end