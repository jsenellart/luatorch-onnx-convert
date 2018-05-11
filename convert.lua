local onnx_pb = require 'onnx_pb'
require('onnx.init')
local convertor = require 'convertors.init'

local path = require('pl.path')

local cmd = torch.CmdLine.new()

local function split(str, sep)
  local res = {}
  local index = 1

  while index <= str:len() do
    local sepStart, sepEnd = str:find(sep, index)

    local sub
    if not sepStart then
      sub = str:sub(index)
      table.insert(res, sub)
      index = str:len() + 1
    else
      sub = str:sub(index, sepStart - 1)
      table.insert(res, sub)
      index = sepEnd + 1
      if index > str:len() then
        table.insert(res, '')
      end
    end
  end

  return res
end

cmd:option('-t7', '', [[Path to the torch serialized file.]])
cmd:option('-require', 'nngraph', [[List of modules to import for loading the torch object (default nngraph).]])
cmd:option('-models', '', [[Field in the object where the models is/are.]])
cmd:option('-output_dir', '', [[Path to directory where onnx models will be serialized.]] ..
                              [[If not set, the extension is changed to _onnxdir.]])
cmd:option('-force', false, [[Force output model creation even if the target file exists.]])

local opt = cmd:parse(arg)

local function convert(output_dir, object, thepath)
  thepath = thepath or ''
  local prefpath = thepath
  if prefpath ~= '' then
    prefpath = prefpath .. '.'
  end

  local tname = convertor.mtype(object)
  if tname == 'table' then
    for k, v in pairs(object) do
      convert(output_dir, v, prefpath..k)
    end
  elseif type(object) == 'userdata' or type(object) == 'table' then
    local convert_func = convertor.isSupported(tname)
    if convert_func then
      print('convert '..thepath..'=`'..tname..'`')
      local graph = convert_func(object)
      if object.output then
        local outputs = object.output
        if type(outputs) ~= 'table' then
          outputs = { outputs }
        end
        for i, o in ipairs(outputs) do
          graph:set_dimension(graph._outputs[i], o:size():totable())
        end
      end
      local model = onnx_pb.ModelProto()
      model.ir_version = onnx_pb.VERSION_IR_VERSION_ENUM.number
      model.producer_name = 'lua-onnx-convert'
      model.producer_version = '0.0.1'
      local version = model.opset_import:add()
      version.version = 6
      model.graph.name = thepath
      graph:build(onnx_pb, model.graph)
      local output = assert(io.open(output_dir .. '/' .. thepath .. '.onnx', "wb"))
      model:SerializeToIOString(output)
      output:close()
    else
      if object.modules and #object.modules == 1 then
        convert(output_dir, object.modules, prefpath..'modules')
      end
      print('\tskipping '..thepath..' ('..tname..')')
    end
  end
end

local function main()
  assert(path.exists(opt.t7), 'file \'' .. opt.t7 .. '\' does not exist.')

  if opt.output_dir:len() == 0 then
    if opt.t7:sub(-3) == '.t7' then
      opt.output_dir = opt.t7:sub(1, -4) -- copy input model without '.t7' extension
    else
      opt.output_dir = opt.t7
    end
    opt.output_dir = opt.output_dir .. '.onnxdir'
  end

  if not opt.force then
    assert(not path.exists(opt.output_dir),
           'output dir already exists; use -force to overwrite.')
  end

  if path.exists(opt.output_dir) then
    assert(path.isdir(opt.output_dir),
             'output ('..opt.output_dir..') is not a directory')
    assert(opt.force,
             'output dir already exists; use -force to overwrite.')
  else
    path.mkdir(opt.output_dir)
  end

  if opt.require ~= '' then
    local requires = split(opt.require, ',')
    for _, r in ipairs(requires) do
      print('import module `'..r..'`')
      require(r)
    end
  end

  -- try loading cutorch modules - while issue warning if not installed
  local _, err = pcall(function()
    require('cutorch')
    require('cunn')
  end)

  if err then
    print('warning: Failed loading cutorch/cunn, GPU models cannot be read')
  end

  print('Loading model \'' .. opt.t7 .. '\'...')

  local obj
  _, err = pcall(function ()
    obj = torch.load(opt.t7)
  end)
  if err then
    error('unable to load the file (' .. err .. ').')
  end

  print('... done.')

  print('Converting model...')
  local models
  if opt.models ~= '' then
    models = obj[opt.models]
  else
    models = obj
  end
  convert(opt.output_dir, models, 'model')
  print('... done.')

end

main()