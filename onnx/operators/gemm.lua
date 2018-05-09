local Gemm, parent = torch.class('onnx.node.Gemm', 'onnx.node')

function Gemm:__init(inputs, outputs, precision,
                     alpha, beta, broadcastC, transposeA, transposeB)
  parent.__init(self, "Gemm", inputs, 3, outputs, 1)
  self._precision = precision
  self._alpha = alpha
  self._beta = beta
  self._broadcastC = broadcastC == 1
  self._transposeA = transposeA == 1
  self._transposeB = transposeB == 1
end

-- given some constraint for the named parameters, check the compatibility
-- and refine these constraints
function Gemm:getShapeConstraint(checker)
  local ca = checker:assert2D(self._inputs[1])
  local cb = checker:assert2D(self._inputs[2])
  local cc = checker:assert1or2D(self._inputs[3])
  local cy = checker:assert2D(self._outputs[1])

  local count = 0
  checker:setChange(true)
  while checker:hasChange() do
    count = count + 1
    checker:setChange(false)
    if not self._transposeA then
      self._pass = checker:dimCheck(ca, 1, cy, 1) or checker:fail()
      if self._transposeB then
        self._pass = checker:dimCheck(ca, 2, cb, 2) or checker:fail()
      else
        self._pass = checker:dimCheck(ca, 2, cb, 1) or checker:fail()
      end
    else
      self._pass = checker:dimCheck(ca, 2, cy, 1) or checker:fail()
      if self._transposeB then
        self._pass = checker:dimCheck(ca, 1, cb, 2) or checker:fail()
      else
        self._pass = checker:dimCheck(ca, 1, cb, 1) or checker:fail()
      end
    end
    if self._transposeB then
      self._pass = checker:dimCheck(cb, 1, cy, 2) or checker:fail()
    else
      self._pass = checker:dimCheck(cb, 2, cy, 2) or checker:fail()
    end
    if #cc == 1 then
      self._pass = checker:dimCheck(cc, 1, cy, 2) or
          checker:dimCheck(cc, 1, cy, 1) or checker:fail()
    elseif #cc == 2 then
      self._pass = checker:dimCheck(cc, 1, cy, 1) or checker:fail()
      self._pass = checker:dimCheck(cc, 2, cy, 2) or checker:fail()
    end
  end

  return count ~= 1
end

function Gemm:build(onnx_pb, node)
  parent.build(self, onnx_pb, node)
  self.addAttribute(node, "alpha", 'f', self._alpha, onnx_pb.AttributeProto.FLOAT)
  self.addAttribute(node, "beta", 'f', self._beta, onnx_pb.AttributeProto.FLOAT)
  self.addAttribute(node, "broadcast", 'i', self._broadcastC and 1 or 0, onnx_pb.AttributeProto.INT)
  self.addAttribute(node, "transA", 'i', self._transposeA and 1 or 0, onnx_pb.AttributeProto.INT)
  self.addAttribute(node, "transB", 'i', self._transposeB and 1 or 0, onnx_pb.AttributeProto.INT)
end