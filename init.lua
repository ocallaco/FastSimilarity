local ffi = require "ffi"
require 'torchffi'

torch.setdefaulttensortype('torch.FloatTensor')

local async = require 'async'

ffi.cdef
[[ 
    typedef void Environment;

    Environment *init(int k, int N, int dim);

    void findClosest(Environment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);
   
    void findClosest2(Environment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

    void findClosestPacked(Environment *environment, unsigned char *matchingSet, float *multipliers, float *queryVector, int *responseSet, float *responseDists);

    void cleanup(Environment *environment);
]]

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

local clib = ffi.load("./fastsimilarity.so")

local N = similarityTable.public_vectors:size(1)
local dim = similarityTable.public_vectors:size(2)
local k = 10

local dataTensor = getDataTensor(similarityTable)

print("starting")
local env = clib.init(k, N, dim)
print("initialized")

local indexes1 = torch.IntTensor(10)
local distances1 = torch.FloatTensor(10)

local indexes2 = torch.IntTensor(10)
local distances2 = torch.FloatTensor(10)

print("running")
for i=N,N-10,-1 do
   local vector = dataTensor[i]

   local sttime = async.hrtime()
   clib.findClosest(env, torch.data(dataTensor), torch.data(vector), torch.data(indexes1), torch.data(distances1))

   local endtime = async.hrtime()

   print("COMPLETED", endtime - sttime)

   for j=1,10 do
      print(indexes1[j],distances1[j])
   end
end

