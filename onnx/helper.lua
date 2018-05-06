local Helper = torch.class('onnx.helper')

onnx_pb = require('onnx_pb')

function Helper.convertPrecision(obj)
  return onnx_pb.TensorProto.FLOAT
end
