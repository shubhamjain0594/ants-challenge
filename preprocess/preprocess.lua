--[[
Preprocessing ANTS dataset to make it
--]]
require 'paths'
require 'image'
require 'csvigo'


local CSVPATH = '/data4/ants-challenge/training_dataset.csv'
local DATAPATH = '/data4/ants-challenge/frames'
local PATCHESPATH = '/data4/ants-challenge/patches'
local PATCHDICTPATH = '/data4/ants-challenge/patch_dict.dat'


--- Returns a table which indexes by frame number and value is list of [ants, x, y] in the frame
function getCSVDataset()
    --- Sets up the CSV
    -- var_1 is ant_id
    -- var_2 is frame_id
    -- var_3 is x-coord
    -- var_4 is y-coord
    -- var_5 is boolean if present or not
    local csv = csvigo.load{path=CSVPATH, header='true'}

    local csvDict = {}
    local antsDict = {}
    antsCount = 1

    for i=2,#csv['var_1'] do
        local antId = tonumber(csv['var_1'][i])
        local frameNum = tonumber(csv['var_2'][i])
        local xCoord = tonumber(csv['var_3'][i])
        local yCoord = tonumber(csv['var_4'][i])
        local presence = tonumber(csv['var_5'][i])
        if not csvDict[frameNum] then
            csvDict[frameNum] = {}
        end
        if not antsDict[antId] then
            antsDict[antId] = antsCount
            antsCount = antsCount + 1
        end
        if presence == 1 then
            local vec = torch.Tensor(3)
            vec[1] = antId
            vec[2] = xCoord
            vec[3] = yCoord
            csvDict[frameNum][#csvDict[frameNum]+1] = vec
        end
    end
    print(("Number of frames %d, Number of Ants %d"):format(#csvDict, antsCount))
    return csvDict, antsDict
end


function createPatches()
    local framesList = {}
    for f in paths.iterfiles(DATAPATH) do
        local frameId = tonumber(paths.basename(f, '.jpeg'))
        framesList[frameId] = paths.concat(DATAPATH, f)
    end
    local csvDict, antDict = getCSVDataset()
    print(antDict)
    local size = 96
    local patchCount = 1
    for key, value in pairs(csvDict) do
        print(("Serving frame %d"):format(key))
        local framePath = framesList[key]
        local img = image.load(framePath)
        local xCorner = 1
        while xCorner < img:size(3) do
            if (xCorner + size - 1) > img:size(3) then
                xCorner = img:size(3) - size + 1
            end
            local yCorner = 1
            while yCorner < img:size(2) do
                if (yCorner + size - 1) > img:size(2) then
                    yCorner = img:size(2) - size + 1
                end
                -- print(xCorner, yCorner)
                local patch = img[{{},{yCorner, yCorner+size-1},{xCorner, xCorner+size-1}}]

                -- Find class of the patch
                local minDist = size*size
                local xCenter = xCorner + size/2
                local yCenter = yCorner + size/2
                local class = 0
                for i=1,#value do
                    if value[i][2] > xCorner and value[i][2] < xCorner + size and value[i][3] > yCorner and value[i][3] < yCorner + size then
                        local xDist = xCenter - value[i][2]
                        local yDist = yCenter - value[i][3]
                        local dist = xDist*xDist + yDist*yDist
                        if dist < minDist then
                            class = antDict[value[i][1]]
                        end
                    end
                end
                local patchFile = paths.concat(PATCHESPATH, tostring(patchCount)..'_'..tostring(class)..'.png')
                print(patchFile)
                image.save(patchFile, patch)
                patchCount = patchCount+1
                yCorner = yCorner + size
            end
            xCorner = xCorner + size
        end
    end
end


function createPatchDict()
    local framesList = {}
    for f in paths.iterfiles(DATAPATH) do
        local frameId = tonumber(paths.basename(f, '.jpeg'))
        framesList[frameId] = paths.concat(DATAPATH, f)
    end
    local csvDict, antDict = getCSVDataset()
    -- print(antDict)
    local size = 96
    local patchCount = 1
    local patchDict = {}
    for key, value in pairs(csvDict) do
        print(("Serving frame %d"):format(key))
        local framePath = framesList[key]
        local img = image.load(framePath)
        local xCorner = 1
        while xCorner < img:size(3) do
            if (xCorner + size - 1) > img:size(3) then
                xCorner = img:size(3) - size + 1
            end
            local yCorner = 1
            while yCorner < img:size(2) do
                if (yCorner + size - 1) > img:size(2) then
                    yCorner = img:size(2) - size + 1
                end
                -- print(xCorner, yCorner)
                local patch = img[{{},{yCorner, yCorner+size-1},{xCorner, xCorner+size-1}}]

                -- Find class of the patch
                local minDist = size*size
                local xCenter = xCorner + size/2
                local yCenter = yCorner + size/2
                local class = 0
                for i=1,#value do
                    if value[i][2] > xCorner and value[i][2] < xCorner + size and value[i][3] > yCorner and value[i][3] < yCorner + size then
                        local xDist = xCenter - value[i][2]
                        local yDist = yCenter - value[i][3]
                        local dist = xDist*xDist + yDist*yDist
                        if dist < minDist then
                            class = antDict[value[i][1]]
                        end
                    end
                end
                patchDict[#patchDict+1] = {framePath, xCorner, yCorner, class}
                patchCount = patchCount+1
                yCorner = yCorner + size
            end
            xCorner = xCorner + size
        end
    end

    torch.save(PATCHDICTPATH, patchDict)
end

createPatchDict()

-- createPatches()
