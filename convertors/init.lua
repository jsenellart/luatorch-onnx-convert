local convertor = {}

-- cache the open convertors file
local convertors = {}

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

function convertor.mtype(object)
  if type(object) == 'table' and object.__typename then
    return object.__typename
  else
    return torch.type(object)
  end
end

function convertor.isSupported(tname)
  local namespace = tname
  local object = ''
  local decomp_name = split(tname, '%.')
  if #decomp_name > 1 then
    namespace = tname:sub(1, -decomp_name[#decomp_name]:len()-2)
    object = tname:sub(-decomp_name[#decomp_name]:len())
  end
  if convertors[namespace] == nil then
    local _, err = pcall(function()
      convertors[namespace] = require('convertors.onnx_'..namespace)
    end)
    if err then
      print('no convertors for '..namespace)
      convertors[namespace] = false
    end
  end
  return convertors[namespace] and convertors[namespace][object]
end

return convertor