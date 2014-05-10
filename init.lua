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

    void findClosest3(Environment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

    void findClosestPacked(Environment *environment, unsigned char *matchingSet, float *multipliers, float *queryVector, int *responseSet, float *responseDists);

    void cleanup(Environment *environment);
]]

local similarity = {}

similarity.init = function(dataTensor, k, N, dim)

   local clib = ffi.load("fastsimilarity")

   local env = clib.init(k, N, dim)

   local indexes = torch.IntTensor(k)
   local distances = torch.FloatTensor(k)

   local finder = {}

   finder.findClosest = function(queryVector)
      clib.findClosest(env, torch.data(dataTensor), torch.data(queryVector), torch.data(indexes), torch.data(distances))
      local response = {}
      for i=1,k do
         if indexes[i] < 0 or indexes[i] > N then break end

         table.insert(response, {indexes[i], distances[i]})
      end
       
      return response
   end

   return finder
end

return similarity

