local onnx_nn = require 'convertors.onnx_nn'
local convertor = require 'convertors.init'

local onnx_onmt = {}

function onnx_onmt.WordEmbedding(obj, nInputs, nonbatch_mode)
  return onnx_nn.LookupTable(obj.net, nInputs, nonbatch_mode)
end

function onnx_onmt.LSTM(obj, nInputs, nonbatch_mode)
  return onnx_nn.gModule(obj.net, nInputs, nonbatch_mode)
end

function onnx_onmt.Bridge(obj, nInputs, nonbatch_mode)
  local obj = obj.net
  local tname = convertor.mtype(obj)
  if type(obj) == 'userdata' or type(obj) == 'table' then
    local convert_func = convertor.isSupported(tname)
    if convert_func then
      return convert_func(obj, nInputs, nonbatch_mode)
    else
      error('module `'..tname..'` not supported')
    end
  else
    error("unsupported module in onmt.Bridge: `"..tname.."`")
  end
end

function onnx_onmt.GlobalAttention(obj, nInputs, nonbatch_mode)
  local obj = obj.net
  local tname = convertor.mtype(obj)
  if type(obj) == 'userdata' or type(obj) == 'table' then
    local convert_func = convertor.isSupported(tname)
    if convert_func then
      return convert_func(obj, nInputs, nonbatch_mode)
    else
      error('module `'..tname..'` not supported')
    end
  else
    error("unsupported module in onmt.Bridge: `"..tname.."`")
  end
end

function onnx_onmt.Encoder(obj, nInputs, nonbatch_mode)
  local obj = obj.network
  local tname = convertor.mtype(obj)
  if type(obj) == 'userdata' or type(obj) == 'table' then
    local convert_func = convertor.isSupported(tname)
    if convert_func then
      return convert_func(obj, nInputs, nonbatch_mode)
    else
      error('module `'..tname..'` not supported')
    end
  else
    error("unsupported module in onmt.Encoder: `"..tname.."`")
  end
end

function onnx_onmt.Decoder(obj, nInputs, nonbatch_mode)
  local obj = obj.network
  local tname = convertor.mtype(obj)
  if type(obj) == 'userdata' or type(obj) == 'table' then
    local convert_func = convertor.isSupported(tname)
    if convert_func then
      return convert_func(obj, nInputs, nonbatch_mode)
    else
      error('module `'..tname..'` not supported')
    end
  else
    error("unsupported module in onmt.Decoder: `"..tname.."`")
  end
end

return onnx_onmt