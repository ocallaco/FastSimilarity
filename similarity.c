#include "similarity.h"
#include <float.h>
#include <stdlib.h>

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

void findClosest(Environment *environment, unsigned char *matchingSet, float *multipliers, 
                                float *queryVector, int *responseSet, float *responseDists){

    clearEnv(environment);
    int dim = environment->dim;

    for(int i = 0; i < environment->N; i++){
        float distance = 0;
        for(int j = 0; j < dim; j++){
            distance += abs(((float)(matchingSet[(i * dim) + j]) * multipliers[i]) - queryVector[j]);
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

