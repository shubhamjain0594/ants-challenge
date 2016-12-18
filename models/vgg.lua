require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

local convolution = nn.SpatialConvolution
local avg = nn.SpatialAveragePooling
local reLU = nn.ReLU
local max = nn.SpatialMaxPooling
local sBatchNorm = nn.SpatialBatchNormalization
local upConvolution = nn.SpatialFullConvolution
local dropout = nn.SpatialDropout

-- building block
local function convBNReLU(nInputPlane, nOutputPlane)
  local block = nn.Sequential()
  block:add(convolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  block:add(sBatchNorm(nOutputPlane,1e-3))
  block:add(reLU(true))
  return block
end

function createModel(opt)
  local nbClasses = opt.nbClasses or 73
  local nbChannels = opt.nbChannels or 3

  local input = nn.Identity()()

  local L1 = convBNReLU(nbChannels,64):add(dropout(0.3))(input)
  local L2 = max(2,2,2,2):ceil()(convBNReLU(64,64)(L1))

  local L3 = convBNReLU(64,128):add(dropout(0.4))(L2)
  local L4 = max(2,2,2,2):ceil()(convBNReLU(128,128)(L3))

  local L5 = convBNReLU(128,256):add(dropout(0.4))(L4)
  local L6 = convBNReLU(256,256):add(dropout(0.4))(L5)
  local L7 = max(2,2,2,2):ceil()(convBNReLU(256,256)(L6))

  local L8 = convBNReLU(256,512):add(dropout(0.4))(L7)
  local L9 = convBNReLU(512,512):add(dropout(0.4))(L8)
  local L10 = max(2,2,2,2):ceil()(convBNReLU(512,512):add(dropout(0.4))(L9))

  local L11 = convBNReLU(512,512):add(dropout(0.4))(L10)
  local L12 = convBNReLU(512,512):add(dropout(0.4))(L11)
  local L13 = max(2,2,2,2):ceil()(convBNReLU(512,512):add(dropout(0.4))(L12))

  local L14 = nn.View(-1, 9*512)(L13)
  local L15 = nn.Sequential():add(nn.Linear(9*512, 500)):add(nn.BatchNormalization(500)):add(nn.ReLU(true)):add(nn.Dropout(0.5))(L14)
  local L16 = nn.Sequential():add(nn.Linear(500, nbClasses)):add(nn.BatchNormalization(nbClasses)):add(nn.ReLU(true)):add(nn.Dropout(0.5))(L15)

  local model = nn.Sequential():add(nn.gModule({input},{L16}))

  -- initialization from MSR
  local function MSRinit(net)
    local function init(name)
      for k,v in pairs(net:findModules(name)) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        v.bias:zero()
      end
    end
    -- have to do for both backends
    init'nn.SpatialConvolution'
  end

  MSRinit(model)

  -- check that we can propagate forward without errors
  -- should get 16x10 tensor
  -- print(model:cuda():forward(torch.CudaTensor(16,3,96,96)):size())

  return model, 'vgg_16'
end

-- createModel({})
