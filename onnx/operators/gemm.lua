local Gemm, parent = torch.class('onnx.node.Gemm', 'onnx.node')

function Gemm:__init(inputs, outputs, precision,
                     alpha, beta, broadcastC, transposeA, transposeB)
  parent.__init(self, "Gemm", inputs, 3, outputs, 1)
  self._precision = precision
  self._alpha = alpha
  self._beta = beta
  self._broadcastC = broadcastC
  self._transposeA = transposeA
  self._transposeB = transposeB
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
  parent.build(self, node)
  self.addAttribute(node, "alpha", 'f', self._alpha, onnx_pb.AttributeProto.FLOAT)
  self.addAttribute(node, "beta", 'f', self._beta, onnx_pb.AttributeProto.FLOAT)
  self.addAttribute(node, "broadcast", 'i', self._broadcastC, onnx_pb.AttributeProto.INT)
  self.addAttribute(node, "transA", 'i', self._transposeA, onnx_pb.AttributeProto.INT)
  self.addAttribute(node, "transB", 'i', self._transposeB, onnx_pb.AttributeProto.INT)
end