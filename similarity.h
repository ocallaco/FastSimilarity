
typedef struct {
    int k;
    int N;
    int dim;
    int *indexes;
    float *distances;
} Environment;

Environment *init(int k, int N, int dim);

void findClosest(Environment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

void findClosest2(Environment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

void findClosest3(Environment *environment, float *matchingSet, float *queryVector, int *responseSet, float *responseDists);

void findClosestPacked(Environment *environment, unsigned char *matchingSet, float *multipliers, 
                                float *queryVector, int *responseSet, float *responseDists);


void cleanup(Environment *environment);
