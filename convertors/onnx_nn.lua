local convertor = require 'convertors.init'
local onnx_nn = require 'convertors.onnx_nngraph'

function onnx_nn.Linear(obj)
  if obj.bias == nil then
    local graph = onnx.graph.new({'x'}, {'y'})
    graph:add_node(onnx.node.Transpose.new({'b'}, {'bt'},
                                           onnx.helper.convertPrecision(obj.weight),
                                           { 1, 0 }))
    graph:add_node(onnx.node.MatMul.new({'x', 'bt'}, {'y'},
                                        onnx.helper.convertPrecision(obj.weight)))
    graph:add_initializer('b', obj.weight)
    return graph
  else
    local graph = onnx.graph.new({'x'}, {'y'})
    graph:add_node(onnx.node.Gemm.new({'x', 'b', 'c'}, {'y'},
                                      onnx.helper.convertPrecision(weight),
                                      1.0, -- alpha
                                      1.0, -- beta
                                      1,   -- broadcast C
                                      0,   -- transpose A
                                      1))  -- transpose B
    graph:add_initializer('b', obj.weight)
    graph:add_initializer('c', obj.bias)
    return graph
  end      
end

function onnx_nn.Identity(obj)
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Identity.new({'x'}, {'y'}))
  return graph
end

function onnx_nn.Tanh(obj)
  local graph = onnx.graph.new({'x'}, {'y'})
  graph:add_node(onnx.node.Tanh.new({'x'}, {'y'}))
  return graph
end

return onnx_nn