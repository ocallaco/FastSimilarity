local ffi = require "ffi"
require 'torchffi'

torch.setdefaulttensortype('torch.FloatTensor')

local async = require 'async'

ffi.cdef
[[ 
    typedef void FloatEnvironment;
    
    typedef void IntEnvironment;

    FloatEnvironment *floatInit(int k, int N, int dim);

    void findClosestFloat(FloatEnvironment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

    void findClosestFloat2(FloatEnvironment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

    void findClosestFloat3(FloatEnvironment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

    void findClosestPacked(FloatEnvironment *environment, unsigned char *matchingSet, float *multipliers, 
    float *queryVector, int *responseSet, float *responseDists);

    void floatCleanup(FloatEnvironment *environment);

    // Int Similarity Functions
    IntEnvironment *intInit(int k, int N, int dim);

    void findClosestInt(IntEnvironment *environment, int *matchingSet, int *queryVector, int *responseSet, int *responseDists);

    void intCleanup(IntEnvironment *environment);
    ]]

local similarity = {}

similarity.floatinit = function(dataTensor, k, N, dim)

   local clib = ffi.load("fastsimilarity")

   local env = clib.floatInit(k, N, dim)

   local indexes = torch.IntTensor(k)
   local distances = torch.FloatTensor(k)

   local finder = {}

   finder.findClosest = function(queryVector)
      clib.findClosestFloat(env, torch.data(dataTensor), torch.data(queryVector), torch.data(indexes), torch.data(distances))
      local response = {}
      for i=1,k do
         if indexes[i] < 0 or indexes[i] > N then break end

         table.insert(response, {indexes[i], distances[i]})
      end

      return response
   end


   finder.findClosest2 = function(queryVector)
      clib.findClosestFloat2(env, torch.data(dataTensor), torch.data(queryVector), torch.data(indexes), torch.data(distances))
      local response = {}
      for i=1,k do
         if indexes[i] < 0 or indexes[i] > N then break end

         table.insert(response, {indexes[i], distances[i]})
      end

      return response
   end
   
   finder.findClosest3 = function(queryVector)
      clib.findClosestFloat3(env, torch.data(dataTensor), torch.data(queryVector), torch.data(indexes), torch.data(distances))
      local response = {}
      for i=1,k do
         if indexes[i] < 0 or indexes[i] > N then break end

         table.insert(response, {indexes[i], distances[i]})
      end

      return response
   end

   finder.findClosestInt = function(queryVector)
      clib.findClosestInt(env, torch.data(dataTensor), torch.data(queryVector), torch.data(indexes), torch.data(distances))
      local response = {}
      for i=1,k do
         if indexes[i] < 0 or indexes[i] > N then break end

         table.insert(response, {indexes[i], distances[i]})
      end

      return response
   end

   return finder
end

similarity.intinit = function(dataTensor, k, N, dim)

   local clib = ffi.load("fastsimilarity")

   local env = clib.intInit(k, N, dim)

   local indexes = torch.IntTensor(k)
   local distances = torch.IntTensor(k)

   local finder = {}

   finder.findClosest = function(queryVector)
      clib.findClosestInt(env, torch.data(dataTensor), torch.data(queryVector), torch.data(indexes), torch.data(distances))
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

