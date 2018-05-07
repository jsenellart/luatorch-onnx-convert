local Helper = torch.class('onnx.helper')

local onnx_pb = require('onnx_pb')

function Helper.convertPrecision(_)
  return onnx_pb.TensorProto.FLOAT
end
