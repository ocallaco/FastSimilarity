#include "similarity.h"
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include "omp.h"
#endif

#include <emmintrin.h>


#define abs(x)  ( ( (x) < 0) ? -(x) : (x) )


#define THFloatVectorDist(x, y, n, ans) {                  \
}



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


// optimized with SSE
void findClosest(Environment *environment, float *matchingSet, 
                                float *queryVector, int *responseSet, float *responseDists){

#ifdef _OPENMP
    long maxthreads = omp_get_max_threads();
    Environment *environments[maxthreads]; 

    //set up data store for each thread
    environments[0] = environment;

    for(int i = 1; i < maxthreads; i++){
        environments[i] = init(environment->k, environment->N, environment->dim);
        clearEnv(environments[i]);
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

        Environment *env = environments[id];

        clearEnv(env);
        int dim = env->dim;

#pragma omp for
        for(int i = 0; i < env->N; i++){
            long startIndex = (long)(i * dim);
            float distance[4];

            // SSE STUFF!
            __m128 vsum = _mm_set1_ps(0.0);                        
            long k;
            
            // I think k should be incrementing should be by 4s, but (possibly due to mem unaligned) it gets the wrong answer unless you do k += 1
            for (k = 0; k < dim; k += 4){                       
                __m128 va = _mm_loadu_ps(matchingSet + startIndex + k);
                __m128 vb = _mm_loadu_ps(queryVector + k);
                __m128 vdiff = _mm_sub_ps(va, vb);
                __m128 vnegdiff = _mm_sub_ps(_mm_set1_ps(0.0), vdiff);
                __m128 vabsdiff = _mm_max_ps(vdiff, vnegdiff);
                vsum = _mm_add_ps(vsum, vabsdiff);
            }
            _mm_storeu_ps(distance,vsum); 

            // ALL DONE!
            //
            addEntry(env, i, distance[0] + distance[1] + distance[2] + distance[3]);
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
                cleanup(environments[x]);
            }
        }
    }
            
    for(int i = 0; i < environment->k; i++){
        responseSet[i] = environment->indexes[i] + 1;
        responseDists[i] = environment->distances[i];
    }

}


// not optimized
void findClosest2(Environment *environment, float *matchingSet, 
                                float *queryVector, int *responseSet, float *responseDists){

#ifdef _OPENMP
    long maxthreads = omp_get_max_threads();
    Environment *environments[maxthreads]; 

    //set up data store for each thread
    environments[0] = environment;

    for(int i = 1; i < maxthreads; i++){
        environments[i] = init(environment->k, environment->N, environment->dim);
        clearEnv(environments[i]);
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

        Environment *env = environments[id];

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
            for (int x = 1; x < nthreads; x++) {
                for(int y = 0; y < env->k; y++){
                    addEntry(environment, environments[x]->indexes[y], environments[x]->distances[y]);
                }
                cleanup(environments[x]);
            }
        }
    }
            
    for(int i = 0; i < environment->k; i++){
        responseSet[i] = environment->indexes[i] + 1;
        responseDists[i] = environment->distances[i];
    }

}


// parallel on inner loop -- doesn't go faster
void findClosest3(Environment *environment, float *matchingSet, 
                                float *queryVector, int *responseSet, float *responseDists){

#ifdef _OPENMP
    long maxthreads = omp_get_max_threads();
#else
    long maxthreads = 1;
#endif


    clearEnv(environment);
    int dim = environment->dim;

    for(int i = 0; i < environment->N; i++){
        float distances[maxthreads];

        for(int j = 0; j < maxthreads; j++){
            distances[j] = 0;
        }
        int startIndex = i * dim;

#pragma omp parallel
        {
            // partial gradients
#ifdef _OPENMP
            long id = omp_get_thread_num();
#else
            long id = 0;
#endif

#pragma omp for
            for(int j = 0; j < dim; j++){
                distances[id] += abs(matchingSet[startIndex + j] - queryVector[j]);
            }


#pragma omp barrier
            if (id==0) {
                float distance = distances[0];
                for(int j = 1; j < maxthreads; j++){
                    distance += distances[j];
                }

                addEntry(environment, i, distance);
            }
        }
    }

    for(int i = 0; i < environment->k; i++){
        responseSet[i] = environment->indexes[i] + 1;
        responseDists[i] = environment->distances[i];
    }

}


void findClosestInt(Environment *environment, int *matchingSet, 
                                int *queryVector, int *responseSet, int *responseDists){

#ifdef _OPENMP
    long maxthreads = omp_get_max_threads();
    Environment *environments[maxthreads]; 

    //set up data store for each thread
    environments[0] = environment;

    for(int i = 1; i < maxthreads; i++){
        environments[i] = init(environment->k, environment->N, environment->dim);
        clearEnv(environments[i]);
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

        Environment *env = environments[id];

        clearEnv(env);
        int dim = env->dim;

#pragma omp for
        for(int i = 0; i < env->N; i++){
            long startIndex = (long)(i * dim);
            int distance = 0;
            for(int j = 0; j < dim; j++){
                distance += abs(matchingSet[startIndex + j] - queryVector[j])
            }

            addEntry(env, i, (float)distance);
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
                cleanup(environments[x]);
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

