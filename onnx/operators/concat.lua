local Concat, parent = torch.class('onnx.node.Concat', 'onnx.node')

function Concat:__init(inputs, outputs, axis)
  parent.__init(self, "Concat", inputs, #inputs, outputs, 1)
  self._axis = axis
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Concat:getShapeConstraint(checker)
  checker:setChange(false)

  local cy = checker:getParam(self._outputs[1])

  local nbdim
  local sizes
  local sumaxisx = 0

  if #cy ~= 0 then
    sizes = cy
    nbdim = #cy
  end
  for _, p in pairs(self._inputs) do
    local cx = checker:getParam(p)
    if #cx ~= 0 then
      assert(nbdim == nil or nbdim == #cx, "inconsistent number of dimensions")
      if sumaxisx ~= nil and cx[self._axis+1] > 0 then
        sumaxisx = sumaxisx + cx[self._axis+1]
      else
        sumaxisx = nil
      end
      nbdim = #cx
      sizes = {}
      for _, v in ipairs(cx) do
        table.insert(sizes, v)
      end
    else
      sumaxisx = nil
    end
  end

  if nbdim ~= nil then
    for _, p in pairs(self._inputs) do
      local cx = checker:assertND(p, nbdim)
      for i = 1, nbdim do
        if i-1 ~= self._axis then
          checker:dimCheck(cx, i, sizes, i)
        end
      end
    end

    if #cy == 0 then
      cy = checker:assertND(self._outputs[1], nbdim)
      checker:setChange(true)
    end
    if sumaxisx ~= nil then
      sizes[self._axis+1] = sumaxisx
    else
      sizes[self._axis+1] = cy[self._axis+1]
    end
    for i = 1, nbdim do
      checker:dimCheck(cy, i, sizes, i)
    end
  end

  return checker:hasChange()
end

function Concat:build(onnx_pb, node)
  parent.build(self, onnx_pb, node)
  self.addAttribute(node, "axis", 'i', self._axis, onnx_pb.AttributeProto.INT)
end