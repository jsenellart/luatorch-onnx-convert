local Checker = torch.class('onnx.checker')

function Checker:__init()
  self._params = {}
  self._types = {}
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

function Checker:setType(p, t)
  self._types[p] = t
end

function Checker:getType(p)
  return self._types[p] or "FLOAT"
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
  for _, S in pairs(self._params) do
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
    self._err = '`'..p1[i1]..'` (dim '..i1..') different from `'..p2[i2]..'` (dim '..i2..')'
    return false
  end
end

function Checker:sameShape(t)
  local idx_nz = 1
  while idx_nz <= #t and #t[idx_nz] == 0 do
    idx_nz = idx_nz + 1
  end
  if idx_nz > #t then
    -- cannot find a non null member
    return true
  end
  for j = 1, #t do
    if #t[j] ~= 0 and #t[j] ~= #t[idx_nz] then
      self._err = 'different shapes: '..#t[idx_nz]..'/'..#t[j]
      return false
    end
    if #t[j] == 0 then
      for _, d in ipairs(t[idx_nz]) do
        table.insert(t[j], d)
      end
      self._change = true
    else
      for h = 1, #t[idx_nz] do
        self:dimCheck(t[idx_nz], h, t[j], h)
      end
    end
  end
  return true
end

function Checker:setDims(p1, dims)
  local p1dim = self._params[p1]
  if p1dim == nil or #p1dim == 0 then
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

function Checker:assertND(param, n)
  if self._params[param] == nil or #self._params[param] == 0 then
    self._change = true
    self._params[param] = { }
    for i = 1, n do
      table.insert(self._params[param], self:getUnkDimIdx())
    end
  else
    assert(#self._params[param] == n, "param `"..param.."` has inconsistent number of dimension "
           ..#self._params[param].."/"..n)
  end
  return self._params[param]
end

function Checker:assert2D(param)
  return self:assertND(param, 2)
end

function Checker:assert1D(param)
  return self:assertND(param, 1)
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