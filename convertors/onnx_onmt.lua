local onnx_nn = require 'convertors.onnx_nn'

local onnx_onmt = {}

function onnx_onmt.WordEmbedding(obj, nInputs)
  return onnx_nn.LookupTable(obj.net, nInputs)
end

function onnx_onmt.LSTM(obj, nInputs)
  return onnx_nn.gModule(obj.net, nInputs)
end

return onnx_onmt