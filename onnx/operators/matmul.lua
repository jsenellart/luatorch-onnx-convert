local MatMul, parent = torch.class('onnx.node.MatMul', 'onnx.node')

function MatMul:__init(inputs, outputs, precision)
  parent.__init(self, "MatMul", inputs, 2, outputs, 1)
  self._precision = precision
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function MatMul:getShapeConstraint(checker)
  checker:setChange(false)

  local ca = checker:getParam(self._inputs[1])
  local cb = checker:getParam(self._inputs[2])
  local cy = checker:getParam(self._outputs[1])

  if #ca > 2 or #cb > 2 or #cy >= 2 then
    local n = #ca
    if #cb > n then n = #cb end
    if #cy > n then n = #cb end
    ca = checker:assertND(self._inputs[1], n)
    cb = checker:assertND(self._inputs[2], n)
    cy = checker:assertND(self._outputs[1], n)
    local b = n - 2
    for i = 1, b do
      checker:dimCheck(ca, i, cb, i)
      checker:dimCheck(ca, i, cy, i)
    end
    self._pass = checker:dimCheck(ca, 1+b, cy, 1+b) or checker:fail()
    self._pass = checker:dimCheck(ca, 2+b, cb, 1+b) or checker:fail()
    self._pass = checker:dimCheck(cb, 2+b, cy, 2+b) or checker:fail()
  elseif #ca == 1 or #cb == 1 or #cy == 1 then
    cy = checker:assert1D(self._outputs[1])
    if #ca == 1 or #cb == 2 then
      ca = checker:assert1D(self._inputs[1])
      cb = checker:assert2D(self._inputs[2])
      self._pass = checker:dimCheck(ca, 1, cb, 1) or checker:fail()
      self._pass = checker:dimCheck(cb, 2, cy, 1) or checker:fail()
    elseif #cb == 1 then
      ca = checker:assert2D(self._inputs[1])
      cb = checker:assert1D(self._inputs[2])
      self._pass = checker:dimCheck(ca, 1, cy, 1) or checker:fail()
      self._pass = checker:dimCheck(ca, 2, cb, 1) or checker:fail()
    end
  end

  return checker:hasChange()
end
