
typedef struct {
    int k;
    int N;
    int dim;
    int *indexes;
    float *distances;
} FloatEnvironment;

typedef struct {
    int k;
    int N;
    int dim;
    int *indexes;
    int *distances;
} IntEnvironment;


// Float Similarity Functions
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

void findClosestInt2(IntEnvironment *environment, int *matchingSet, int *queryVector, int *responseSet, int *responseDists);

void intCleanup(IntEnvironment *environment);

