--[[
AntsDataset class for reading through BRATS Dataset
--]]
require 'paths'
require 'image'
require 'csvigo'

local tnt = require 'torchnet'
torch.setdefaulttensortype('torch.FloatTensor')

local AntsDataset = torch.class('tnt.AntsDataset','tnt.Dataset',tnt)

AntsDataset.__init = function(self, opt)
    self.datapath = opt.datapath or '/data4/ants-challenge/frames'
    self.csvpath = opt.csvpath or '/data4/ants-challenge/training_dataset.csv'
    self.patchDictPath = '/data4/ants-challenge/patch_dict.dat'
    self.trainBatchSize = opt.trainBatchSize or 32
    self.valBatchSize = opt.valBatchSize or 32
    self.mode = opt.mode
    self.imsize = 96
    self.stride = 96

    self.valparam = 10 -- k-fold validation
    self.dsize = 0

    self:setup(opt)
end


function AntsDataset:setup(opt)
    -- print(opt)
    local patchDict = torch.load(self.patchDictPath)
    local idxKeyDict = {}
    local keyPatchDict = {}
    for i=1,#patchDict do
        local key = tonumber(paths.basename(patchDict[i][1], '.jpeg'))
        local class = patchDict[i][4]
        if (self.mode == 'train' and key%self.valparam ~= 0) or (self.mode == 'test' and key%self.valparam == 0) then
            if not keyPatchDict[key] then
                idxKeyDict[#idxKeyDict+1] = key
                keyPatchDict[key] = {}
            end
            if class == 0 then
                if torch.uniform() < 0.05 then
                    keyPatchDict[key][#keyPatchDict[key]+1] = patchDict[i]
                    self.dsize = self.dsize + 1
                end
            else
                keyPatchDict[key][#keyPatchDict[key]+1] = patchDict[i]
                self.dsize = self.dsize + 1
            end
        end
        if opt.size and self.dsize >= opt.size then
            break
        end
    end
    -- print("Dataloader setup done")
    self.idxKeyDict = idxKeyDict
    self.keyPatchDict = keyPatchDict

    --- Setting variables for getting image
    -- print(self.dsize, opt.size)
    self.permIdx = tnt.transform.randperm(#self.idxKeyDict) -- For permutation among the frames
    self.loadedImageIdx = 0 -- location of current image
    self.currIdx = 0 -- location of current index in permutation
    self.loadedPatchIdx = 0 -- location of current patch in the loaded image
end

AntsDataset.size = function(self)
    return self.dsize
end

AntsDataset.get = function(self,idx)
    -- print(idx)
    -- local start = os.clock()
    while not self.loadedImage do
        self.currIdx = self.currIdx + 1
        self.loadedImageIdx = self.permIdx(self.currIdx)
        if #self.keyPatchDict[self.idxKeyDict[self.loadedImageIdx]] > 0 then
            self.loadedPatchIdx = 1
            self.loadedImage = image.load(self.keyPatchDict[self.idxKeyDict[self.loadedImageIdx]][1][1])
            break
        end
    end
    local patchDict = self.keyPatchDict[self.idxKeyDict[self.loadedImageIdx]][self.loadedPatchIdx]
    local framePath = patchDict[1]
    local xCorner = patchDict[2]
    local yCorner = patchDict[3]
    local class = patchDict[4]+1

    local patch = self.loadedImage[{{},{yCorner, yCorner+self.imsize-1},{xCorner, xCorner+self.imsize-1}}]
    local target = torch.Tensor(1)
    target[1] = class + 1

    self.loadedPatchIdx = self.loadedPatchIdx + 1
    if self.loadedPatchIdx > #self.keyPatchDict[self.idxKeyDict[self.loadedImageIdx]] then
        self.loadedImage = nil
    end

    --- Condition for end of dataset and hence reset
    if idx == self.dsize then
        self.permIdx = tnt.transform.randperm(#self.idxKeyDict) -- For permutation among the frames
        self.loadedImageIdx = 0 -- location of current image
        self.currIdx = 0 -- location of current index in permutation
        self.loadedPatchIdx = 0 -- location of current patch in the loaded image
    end

    return { input = patch:div(255):float(), target = target }
    -- end
end

local function test()
    local a = tnt.AntsDataset({mode='train', size=100})
    local start = os.clock()
    print(a.get(a,1), a.size(a))
    print(os.clock()-start)
end

-- test()
