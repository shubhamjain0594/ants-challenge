require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
local tnt = require 'torchnet'

local Machine = torch.class 'Machine'

--- Class that sets engine, criterion, model
-- @param opt
function Machine:__init(opt)
   opt = opt or {}

   self.trainSize = opt.trainSize -- size of training dataset to be used
   self.valSize = opt.valSize -- size of validation dataset to be used

   self.trainBatchSize = opt.trainBatchSize or 1
   self.valBatchSize = opt.valBatchSize or 1

   self.model, self.modelName = self:loadModel(opt) -- model to be used
   self.criterion = self:loadCriterion(opt) -- criterion to be used
   self.engine = self:loadEngine(opt) -- engine to be used

   self.savePath = opt.savePath -- path where models has to be saved
   self.maxepoch = opt.maxepoch -- maximum number of epochs for training
   self.learningalgo = opt.optimMethod -- name of the learning algorithm used
   self.learningRate = opt.learningRate -- learning rate to begin with

   self.tieWeights = opt.tieWeights -- Whether to tie weights

   self.meters = self:loadMeters(opt) -- table of meters, key is the name of meter and value is the meter
   self:attachHooks(opt)
   self:setupEngine(opt)
   self.opt = opt
end

--- Loads the model
-- @return Model loaded in CUDA,Name of the model
function Machine:loadModel(opt)
   local model = opt.model
   dofile(model)
   local net, name
   if not opt.resumeModel then
      net,name = createModel(opt)
   else
      net = torch.load(opt.resumeModel)
      name = 'vgg_16'
   end
   net = net:cuda()
   return net,name
end

--- Loads the criterion
-- @return Criterion loaded in CUDA
function Machine:loadCriterion(opt)
   local criterion = nn.CrossEntropyCriterion()
   criterion = criterion:cuda()
   return criterion
end

--- Loads the engine
-- @return Optim Engine Instance
function Machine:loadEngine(opt)
   local engine = tnt.OptimEngine()
   return engine
end

--- Loads all the meters
-- @return Table of meters such that, key is a string denoting meter name and value is the meter
-- Keys - Training Loss, Training Dice Score, Validation, Validation Dice Score, Param Norm, GradParam Norm, Norm Ratio, Time
function Machine:loadMeters(opt)
   local meters = {}
   meters['Training Loss'] = tnt.AverageValueMeter()
   meters['Training Accuracy'] = tnt.AverageValueMeter()
   meters['Validation Loss'] = tnt.AverageValueMeter()
   meters['Validation Accuracy'] = tnt.AverageValueMeter()
   meters['Param Norm'] = tnt.AverageValueMeter()
   meters['GradParam Norm'] = tnt.AverageValueMeter()
   meters['Norm Ratio'] = tnt.AverageValueMeter()
   meters['Time'] = tnt.TimeMeter()
   return meters
end

--- Resets all the meters
function Machine:resetMeters()
   for i,v in pairs(self.meters) do
      v:reset()
   end
end

--- Prints the values in all the meters
function Machine:printMeters()
   for i,v in pairs(self.meters) do
      io.write(('%s : %.5f | '):format(i,v:value()))
   end
   io.write('\n')
end

--- Trains the model
function Machine:train(opt)
   local engineReq = {}
   engineReq.batchsize = self.trainBatchSize
   engineReq.size = self.trainSize
   engineReq.factordown = self.opt.factordown
   engineReq.mode = 'train'
   self.engine:train{
      network   = self.model,
      iterator  = getIterator(engineReq),
      criterion = self.criterion,
      optimMethod = self.optimMethod,
      config = self.optimConfig,
      maxepoch = self.maxepoch
   }
end

--- Test the model against validation data
function Machine:test(opt)
   local engineReq = {}
   engineReq.batchsize = self.valBatchSize
   engineReq.size = self.valSize
   engineReq.factordown = self.opt.factordown
   engineReq.mode = 'test'
   self.engine:test{
      network   = self.model,
      iterator  = getIterator(engineReq),
      criterion = self.criterion,
   }
end

--- Given the state, it will save the model as ModelName_DatasetName_LearningAlgorithm_epoch_torchnet_EpochNum.t7
function Machine:saveModels(state)
   if state.epoch%2 == 0 then
      local savePath = paths.concat(self.savePath, ('%s_%s_%s_epoch_torchnet_%d.t7'):format(self.modelName,self.learningalgo,self.opt.criterion,state.epoch))
      torch.save(savePath,state.network:clearState())
   end
end

--- Adds hooks to the engine
-- state is a table of network, criterion, iterator, maxEpoch, optimMethod, sample (table of input and target),
-- config, optim, epoch (number of epochs done so far), t (number of samples seen so far), training (boolean denoting engine is in training or not)
-- https://github.com/torchnet/torchnet/blob/master/engine/optimengine.lua for position of hooks as to when they are called
function Machine:attachHooks(opt)

   --- Gets the size of the dataset or number of iterations
   local onStartHook = function(state)
      state.numbatches = state.iterator:execSingle('size')  -- for ParallelDatasetIterator
      state.confusion = optim.ConfusionMatrix(opt.nbClasses)
      state.confusion:zero()
   end

   --- Resets all the meters
   local onStartEpochHook = function(state)
      if self.learningalgo == 'sgd' then
         state.optim.learningRate = self:learningRateScheduler(state,state.epoch+1)
         print(("Epoch : %d, Learning Rate : %.5f "):format(state.epoch+1,state.optim.learningRate or state.config.learningRate))
      end
      print(("Epoch : %d"):format(state.epoch+1))
      self:resetMeters()
      state.confusion:zero()
   end

   --- Transfers input and target to cuda
   local onSampleHook = function(state)
      state.sample.input  = state.sample.input:cuda()
      state.sample.target = state.sample.target
      -- print("Target size : ", state.sample.target:size())
      -- print("Input size : ", state.sample.input:size())
   end

   local onForwardHook = function(state)
      -- print(state.sample.target)
      -- print(state.network.output)
   end

   --- Updates losses and dice score
   local onForwardCriterionHook = function(state)
      -- print("Forward")
      -- print(state.sample.target:size())
      -- print(state.network.output:size())
      state.confusion:batchAdd(state.network.output, state.sample.target)
      if state.training then
         self.meters['Training Loss']:add(state.criterion.output)
      else
         self.meters['Validation Loss']:add(state.criterion.output)
      end
   end

   local onBackwardCriterionHook = function(state)
      -- print(state.criterion.gradInput:norm())
   end

   local onBackwardHook = function(state)
   end

   --- Update the parameter norm, gradient parameter norm, norm ratio and update progress bar to denote number of batches done
   local onUpdateHook = function(state)
      self.meters['Param Norm']:add(state.params:norm())
      self.meters['GradParam Norm']:add(state.gradParams:norm())
      self.meters['Norm Ratio']:add(state.gradParams:norm()/state.params:norm())
      xlua.progress(state.t,state.numbatches)
   end

   --- Sets t to 0, does validation and prints results of the epoch
   local onEndEpochHook = function(state)
      -- print("Epoch done")
      if state.training then
         state.confusion:updateValids()
         self.meters['Training Accuracy']:add(state.confusion.totalValid*100)
      end
      state.t = 0
      self:test()
      self:printMeters()
      self:saveModels(state)
   end

   local onEndHook = function(state)
      if not state.training then
         state.confusion:updateValids()
         self.meters['Validation Accuracy']:add(state.confusion.totalValid*100)
      end
   end

   --- Attaching all the hooks
   self.engine.hooks.onStart = onStartHook
   self.engine.hooks.onStartEpoch = onStartEpochHook
   self.engine.hooks.onSample = onSampleHook
   self.engine.hooks.onForward = onForwardHook
   self.engine.hooks.onForwardCriterion = onForwardCriterionHook
   self.engine.hooks.onBackwardCriterion = onBackwardCriterionHook
   self.engine.hooks.onBackward = onBackwardHook
   self.engine.hooks.onUpdate = onUpdateHook
   self.engine.hooks.onEndEpoch = onEndEpochHook
   self.engine.hooks.onEnd = onEndHook
end

--- Returns the learning for the epoch
-- @param state State of the training
-- @param epoch Current epoch number
-- @return Learning Rate
-- Training scheduler that reduces learning by factor of 10 rate after every 40 epochs
function Machine:learningRateScheduler(state,epoch)
    local decay = 0
    local step = 25
    decay = math.ceil((epoch/ step))-1
    return self.learningRate*math.pow(0.1, decay)
end

--- Sets up the optim engine based on parameter received
-- @param opt It must contain optimMethod
function Machine:setupEngine(opt)
   if opt.optimMethod=='sgd' then
      self.optimMethod = optim.sgd
      self.optimConfig = {
         learningRate = 0.1,
         momentum = 0.9,
         nesterov = true,
         weightDecay = 0.0001,
         dampening = 0.0,
      }
   elseif opt.optimMethod=='adam' then
      self.optimMethod = optim.adam
      self.optimConfig = {
         learningRate = 0.1
      }
   elseif opt.optimMethod=='adadelta' then
      self.optimMethod = optim.adadelta
      self.optimConfig = {}
   end
end

--- Iterator for moving over data
-- @param opt Contains all information required to generate the iterator
-- opt.transform - Transformation function to be used for data augmentation
-- opt.batchsize - Batch Size of the batch for dataset
-- opt.images - Images to be used in dataset
-- opt.masks - Masks to be used in datase
function getIterator(opt, mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      closure = function()
         local tnt = require 'torchnet'
         dofile('modules/AntsDataset.lua')
         local dataset = tnt.AntsDataset(opt)
         return tnt.BatchDataset{
            dataset = dataset,
            batchsize = opt.batchsize,
         }
      end
   }
end
