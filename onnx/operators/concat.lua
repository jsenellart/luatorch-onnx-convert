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

  local cxs = {}
  local nbdim
  local sizes

  if #cy ~= 0 then
    sizes = cy
    nbdim = #cy
  end

  for _, p in pairs(self._inputs) do
    local cx = checker:getParam(p)
    if #cx ~= 0 then
      assert(nbdim == nil or nbdim == #cx, "inconsistent number of dimensions")
      nbdim = #cx
      sizes = cx
    end
  end

  for _, p in pairs(self._inputs) do
    local cx = checker:getParam(p)
    if #cx ~= 0 then
      assert(nbdim == nil or nbdim == #cx, "inconsistent number of dimensions")
      nbdim = #cx
      sizes = cx
    elseif nbdim ~= nil then
      for i = 1, nbdim do
        table.insert(cx, checker:getUnkDimIdx())
      end
      checker:setChange(true)
    end
    table.insert(cxs, cx)
  end

  if nbdim ~= nil then
    if #cy == 0 then
      for i = 1, nbdim do
        table.insert(cy, checker:getUnkDimIdx())
        checker:dimCheck(cy, i, sizes, i)
      end
      checker:setChange(true)
    end      
    -- size of concatenated dimension
    local height = 0

    for i, cx in pairs(cxs) do
      for j = 1, #cx do
        checker:dimCheck(cx, j, sizes, j)
      end
      if height ~= nil and cx[self._axis+1] > 0 then
        height = height + cx[self._axis+1]
      end
    end
  end

  if height ~= nil then
    if cy[self._axis+1] < 0 then
      self:changeUnk(cy[self._axis+1], height)
    else
      assert(cy[self._axis+1] == height, "inconsistent height of concatenated tensor")
    end
  end

  return checker:hasChange()
end

function Concat:build(onnx_pb, node)
  parent.build(self, onnx_pb, node)
  self.addAttribute(node, "axis", 'i', self._axis, onnx_pb.AttributeProto.INT)
end