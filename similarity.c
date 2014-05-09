#include "similarity.h"
#include <float.h>
#include <stdlib.h>

#ifdef _OPENMP
#include "omp.h"
#endif


#define abs(x)  ( ( (x) < 0) ? -(x) : (x) )

Environment *init(int k, int N, int dim){

    Environment *environment = malloc(sizeof(Environment));
    environment->k = k;
    environment->N = N;
    environment->dim = dim;
    environment->indexes = malloc(sizeof(int) * k);
    environment->distances = malloc(sizeof(float) * k);

    return environment;
}

static inline void addEntry(Environment *environment, int index, float distance){
    int i = 0;
    int tempIndex, newIndex = -1;
    float tempDistance, newDistance = FLT_MAX;

    int *indexes = environment->indexes;
    float *distances= environment->distances;
    
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

static inline void clearEnv(Environment *environment){
    for(int i = 0; i < environment->k; i++){
        environment->indexes[i] = -1;
        environment->distances[i] = FLT_MAX;
    }
}

void findClosest(Environment *environment, float *matchingSet, 
                                float *queryVector, int *responseSet, float *responseDists){

#ifdef _OPENMP
    long maxthreads = omp_get_max_threads();
    Environment *environments[maxthreads]; 

    //set up data store for each thread
    for(int i = 0; i < maxthreads; i++){
        environments[i] = init(environment->k, environment->N, environment->dim)
    }

#else
    long maxthreads = 1;
    Environment *environments[1] 
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

        Environment *env = environments[id]

        clearEnv(env);
        int dim = env->dim;

#pragma omp for
        for(int i = 0; i < env->N; i++){
            float distance = 0;
            int startIndex = i * dim;
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
            if(nthreads > 1){
                for (int x = 0; x < nthreads; x++) {
                    for(int y = 0; y < env->k; y++){
                        addEntry(environment, environments[x]->indexes[y], environments[x]->distances[y])
                    }
                    cleanup(environments[x])
                }
            }
        }
    }
            
    for(int i = 0; i < environment->k; i++){
        responseSet[i] = environment->indexes[i] + 1;
        responseDists[i] = environment->distances[i];
    }

}



void findClosestPacked(Environment *environment, unsigned char *matchingSet, float *multipliers, 
                                float *queryVector, int *responseSet, float *responseDists){

    clearEnv(environment);
    int dim = environment->dim;

    for(int i = 0; i < environment->N; i++){
        float distance = 0;
        int startIndex = i * dim;
        for(int j = 0; j < dim; j++){
            distance += abs(((float)(matchingSet[startIndex + j]) * multipliers[i]) - queryVector[j]);
        }
        addEntry(environment, i, distance);
    }

    for(int i = 0; i < environment->k; i++){
        responseSet[i] = environment->indexes[i] + 1;
        responseDists[i] = environment->distances[i];
    }
}

void cleanup(Environment *environment){
    free(environment->indexes);
    free(environment->distances);
    free(environment);
}

