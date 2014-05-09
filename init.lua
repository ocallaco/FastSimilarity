local ffi = require "ffi"
require 'torchffi'

ffi.cdef
[[ 
    typedef void Environment;

    Environment *init(int k, int N, int dim);

    void findClosest(Environment *environment, unsigned char *matchingSet, float *multipliers, float *queryVector, int *responseSet, float *responseDists);

    void cleanup(Environment *environment);
]]

-- options
local opt = lapp([[
Starts a daemon by type
-f, --file (default './data/SimilarityTable.1.m')
]])

print("loading table")
local similarityTable = torch.load(opt.file)
print("done")

local clib = ffi.load("./fastsimilarity.so")

local N = similarityTable.public_vectors:size(1)
local dim = similarityTable.public_vectors:size(2)
local k = 10

local env = clib.init(k, N, dim)

local indexes = torch.IntTensor(10)
local distances = torch.FloatTensor(10)

for i=N,N-50,-1 do
   local vector = similarityTable.public_vectors[i] * similarityTable.public_multipliers[i]
   
   clib.findClosest(env, torch.data(env, similarityTable.public_vectors), torch.data(similarityTable.public_multipliers), torch.data(vector), torch.data(indexes), torch.data(distances))

   for j=1,10 do
      print(indexes[j],distances[j])
   end
end

