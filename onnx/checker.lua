local Checker = torch.class('onnx.checker')

function Checker:__init()
  self._params = {}
  self._change = false
  self._unkDimIdx = -1;
end

function Checker:setChange(v)
  self._change = v
end

function Checker:hasChange()
  return self._change
end

function Checker:getUnkDimIdx()
  self._unkDimIdx = self._unkDimIdx - 1
  return self._unkDimIdx + 1
end

function Checker:fail()
  error(self._err)
end

function Checker:params()
  return self._params
end

-- get or create a param, we don't know dimension
function Checker:getParam(param)
  if self._params[param] == nil then
    self._change = true
    self._params[param] = {}
  end
  return self._params[param]
end

function Checker:changeUnk(v1, v2)
  self._change = true
  for p, S in pairs(self._params) do
    for i, d in ipairs(S) do
      if d == v1 then
        S[i] = v2
      end
    end
  end
end

function Checker:dimCheck(p1, i1, p2, i2)
  if p1[i1] == p2[i2] then
    return true
  end
  if p1[i1] < 0 then
    self:changeUnk(p1[i1], p2[i2])
    return true
  elseif p2[i2] < 0 then
    self:changeUnk(p2[i2], p1[i1])
    return true
  else
    self._err = '`'..p1..'` (dim '..i1..') different from `'..p2..'` (dim '..i2..')'
    return false
  end
end

function Checker:sameShape(t)
  local i = 1
  while i <= #t and #t[i] == 0 do
    i = i + 1
  end
  if i > #t then
    return true 
  end
  for j = 1, #t do
    if #t[j] ~= 0 and t[j] ~= t[i] then
      self._err = '`'..p1..'` (dim '..i1..') different from `'..p2..'` (dim '..i2..')'
      return false
    end
    if #t[j] == 0 then
      for _, d in ipairs(t[i]) do
        table.insert(t[j], d)
      end
      self._change = true
    end
  end
  return true
end

function Checker:setDims(p1, dims)
  local p1dim = self._params[p1]
  if p1dim == nil then
    self._params[p1] = dims
    self._change = true
    return true
  end
  for i, d in ipairs(p1dim) do
    if dims[i] ~= d then
      if d < 0 then
        self:changeUnk(d, dims[i])
      elseif dims[i] < 0 then
        self:changeUnk(dims[i], d)
      else
        error('incompatible dimension setting')
      end
    end
  end
end

function Checker:assert2D(param)
  if self._params[param] == nil or #self._params[param] == 0 then
    self._change = true
    self._params[param] = { self:getUnkDimIdx(), self:getUnkDimIdx() }
  else
    assert(#self._params[param] == 2, "param `"..param.."` has inconsistent number of dimension")
  end
  return self._params[param]
end

function Checker:assert1D(param)
  if self._params[param] == nil or #self._params[param] == 0 then
    self._change = true
    self._params[param] = { self:getUnkDimIdx() }
  else
    assert(#self._params[param] == 1, "param `"..param.."` has inconsistent number of dimension")
  end
  return self._params[param]
end

function Checker:assert1or2D(param)
  if self._params[param] == nil or #self._params[param] == 0 then
    return {}
  else
    assert(#self._params[param] == 2 or #self._params[param] == 1,
           "param `"..param.."` has inconsistent number of dimension")
  end
  return self._params[param]
end