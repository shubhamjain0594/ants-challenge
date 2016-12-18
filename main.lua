require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
tnt = require 'torchnet'

torch.setnumthreads(1) -- speed up
torch.setdefaulttensortype('torch.FloatTensor')

-- command line instructions reading
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Fortis Data Tumor Classification')
cmd:text()
cmd:text('Options:')
cmd:option('-model','models/vgg.lua','Path of the model to be used')
cmd:option('-trainSize',-1,'Size of the training dataset to be used - -1 for complete')
cmd:option('-valSize',-1,'Size of the validation dataset to be used - -1 for complete')
cmd:option('-trainBatchSize',160,'Size of the batch to be used for training')
cmd:option('-valBatchSize',160,'Size of the batch to be used for validation')
cmd:option('-savePath','data/models/all/','Path to save models')
cmd:option('-optimMethod','sgd','Algorithm to be used for learning - sgd | adam| adadelta')
cmd:option('-maxepoch',250,'Epochs for training')
cmd:option('-criterion','mse','Criterion to be used with the model - mse')
cmd:option('-learningRate',0.1,'Learning rate to begin with')
cmd:option('-randomSeed',1234,'Random seed to be used')
cmd:option('-nbChannels',3,'Number of channels')
cmd:option('-nbClasses',73,'Number of classes')
cmd:option('-resumeModel','data/models/all/vgg_16_adadelta_mse_epoch_torchnet_22.t7','Model to resume if any')


--- Main execution script
function main(opt)
   torch.manualSeed(opt.randomSeed)
   if opt.trainSize <= 0 then
      opt.trainSize = nil
   end
   if opt.valSize <= 0 then
      opt.valSize = nil
   end
   print(opt)

   require 'machine.lua'
   local m = Machine(opt)
   m:train(opt)
end

local opt = cmd:parse(arg or {}) -- Table containing all the above options
main(opt)
