local ffi = require "ffi"
require 'torchffi'

torch.setdefaulttensortype('torch.FloatTensor')

local async = require 'async'
local similarity = require './'

-- options
local opt = lapp([[
Starts a daemon by type
-f, --file (default './data/SimilarityTable.1.m')
]])


local getDataTensor = function(similarityTable)
   local N = similarityTable.public_vectors:size(1)
   local dim = similarityTable.public_vectors:size(2)

   local data_tensor = torch.FloatTensor(dim * N):copy(similarityTable.public_vectors):resize(similarityTable.public_vectors:size())
   local multipliers = torch.Tensor(similarityTable.public_multipliers)

   multipliers:resize(N,1)

   data_tensor:cmul(multipliers:expandAs(data_tensor))
   
   return data_tensor, similarityTable
end


print("loading table")
local similarityTable = torch.load(opt.file)
print("done")

local N = similarityTable.public_vectors:size(1)
local dim = similarityTable.public_vectors:size(2)
local k = 10

local dataTensor = getDataTensor(similarityTable)

print("starting")
local simFinder = similarity.init(dataTensor, k)
print("initialized")

print("running")
for i=N,N-10,-1 do
   local vector = dataTensor[i]

   local sttime = async.hrtime()
   local response = simFinder.findClosest(vector)
   local endtime = async.hrtime()

   print("COMPLETED", endtime - sttime)

   for j,response in ipairs(response) do 
      print(response[1], response[2])
   end
end

