#include "similarity.h"
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include "omp.h"
#endif

#include <emmintrin.h>


#define abs(x)  ( ( (x) < 0) ? -(x) : (x) )


IntEnvironment *intInit(int k, int N, int dim){

    IntEnvironment *environment = malloc(sizeof(IntEnvironment));
    environment->k = k;
    environment->N = N;
    environment->dim = dim;
    environment->indexes = malloc(sizeof(int) * k);
    environment->distances = malloc(sizeof(int) * k);

    return environment;
}

static inline void addEntry(IntEnvironment *environment, int index, int distance){
    int i = 0;
    int tempIndex, newIndex = -1;
    int tempDistance, newDistance = INT_MAX;

    int *indexes = environment->indexes;
    int *distances= environment->distances;
    
    for(i = 0; i < environment->k; i++){
        if(distances[i] > distance){
            newIndex = index;
            newDistance = distance;
            break;
        }
    }

    if(newIndex >= 0){
        do 
        {
            tempIndex = indexes[i];
            tempDistance = distances[i];
            indexes[i] = newIndex;
            distances[i] = newDistance;
            newIndex = tempIndex;
            newDistance = tempDistance;
            i++;
        } while(i < environment->k && newIndex >= 0);
    }
}

static inline void clearEnv(IntEnvironment *environment){
    for(int i = 0; i < environment->k; i++){
        environment->indexes[i] = -1;
        environment->distances[i] = INT_MAX;
    }
}


void findClosestInt(IntEnvironment *environment, int *matchingSet, 
                                int *queryVector, int *responseSet, int *responseDists){

#ifdef _OPENMP
    long maxthreads = omp_get_max_threads();
    IntEnvironment *environments[maxthreads]; 

    //set up data store for each thread
    environments[0] = environment;

    for(int i = 1; i < maxthreads; i++){
        environments[i] = intInit(environment->k, environment->N, environment->dim);
        clearEnv(environments[i]);
    }

#else
    long maxthreads = 1;
    IntEnvironment *environments[1];
    environments[0] = environment;
#endif

#pragma omp parallel
    {
        // partial gradients
#ifdef _OPENMP
        long id = omp_get_thread_num();
#else
        long id = 0;
#endif

        IntEnvironment *env = environments[id];

        clearEnv(env);
        int dim = env->dim;

#pragma omp for
        for(int i = 0; i < env->N; i++){
            long startIndex = (long)(i * dim);

            int distance[4];

            // SSE STUFF!
            __m128i vsum = _mm_set1_epi32(0);                        
            long k;
            
            // I think k should be incrementing should be by 4s, but (possibly due to mem unaligned) it gets the wrong answer unless you do k += 1
            for (k = 0; k < dim; k += 4){                       
                __m128i va = _mm_loadu_si128(matchingSet + startIndex + k);
                __m128i vb = _mm_loadu_si128(queryVector + k);
                __m128i vdiff = _mm_sub_epi32(va, vb);
                __m128i vnegdiff = _mm_sub_epi32(_mm_set1_epi32(0), vdiff);
                __m128i vabsdiff = _mm_max_epi32(vdiff, vnegdiff);
                vsum = _mm_add_epi32(vsum, vabsdiff);
            }
            _mm_storeu_si128(distance,vsum); 

            // ALL DONE!
            //
            addEntry(env, i, distance[0] + distance[1] + distance[2] + distance[3]);

            addEntry(env, i, distance);
        }

        // reduce
#pragma omp barrier
        if (id==0) {
#ifdef _OPENMP
            long nthreads = omp_get_num_threads();
#else
            long nthreads = 1;
#endif
            for (int x = 1; x < nthreads; x++) {
                for(int y = 0; y < env->k; y++){
                    addEntry(environment, environments[x]->indexes[y], environments[x]->distances[y]);
                }
                intCleanup(environments[x]);
            }
        }
    }
            
    for(int i = 0; i < environment->k; i++){
        responseSet[i] = environment->indexes[i] + 1;
        responseDists[i] = environment->distances[i];
    }

}



void findClosestInt2(IntEnvironment *environment, int *matchingSet, 
                                int *queryVector, int *responseSet, int *responseDists){

#ifdef _OPENMP
    long maxthreads = omp_get_max_threads();
    IntEnvironment *environments[maxthreads]; 

    //set up data store for each thread
    environments[0] = environment;

    for(int i = 1; i < maxthreads; i++){
        environments[i] = intInit(environment->k, environment->N, environment->dim);
        clearEnv(environments[i]);
    }

#else
    long maxthreads = 1;
    IntEnvironment *environments[1];
    environments[0] = environment;
#endif

#pragma omp parallel
    {
        // partial gradients
#ifdef _OPENMP
        long id = omp_get_thread_num();
#else
        long id = 0;
#endif

        IntEnvironment *env = environments[id];

        clearEnv(env);
        int dim = env->dim;

#pragma omp for
        for(int i = 0; i < env->N; i++){
            long startIndex = (long)(i * dim);
            int distance = 0;
            for(int j = 0; j < dim; j++){
                distance += abs(matchingSet[startIndex + j] - queryVector[j]);
            }

            addEntry(env, i, distance);
        }

        // reduce
#pragma omp barrier
        if (id==0) {
#ifdef _OPENMP
            long nthreads = omp_get_num_threads();
#else
            long nthreads = 1;
#endif
            for (int x = 1; x < nthreads; x++) {
                for(int y = 0; y < env->k; y++){
                    addEntry(environment, environments[x]->indexes[y], environments[x]->distances[y]);
                }
                intCleanup(environments[x]);
            }
        }
    }
            
    for(int i = 0; i < environment->k; i++){
        responseSet[i] = environment->indexes[i] + 1;
        responseDists[i] = environment->distances[i];
    }

}

void intCleanup(IntEnvironment *environment){
    free(environment->indexes);
    free(environment->distances);
    free(environment);
}

