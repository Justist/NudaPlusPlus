//Enables the use of assert()
#include <cassert>
//To be able to use nvprof. Also needs cudaProfilerStop() at the end
#include <cuda_profiler_api.h>
//Included to create random numbers in cuda functions
#include <curand.h>
#include <curand_kernel.h>
//We occasionally might need to print things
#include <iostream>
//Used for fabs and other float math functions
//#include <math.h>
//Enables the use of vectors
#include <vector>

/*
 * Checks if there is an error made by cuda which isn't shown to us.
 * Use: cudaCheckErrors("<Message>");
 */
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

/*
 * Roughly the same function as layerInit(), however it only applies
 * to the first layer (maybe change this to one layer and use it for
 * scaling as mentioned in the commentary at layerInit()?).
 * This is useful as the first nodes layer needs to contain the input
 * of the network, so those values can be propagated.
 */
__global__
void firstLayerInit(const unsigned int firstNodes, float *values, float *firstLayer) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (unsigned int i = index; i < firstNodes; i += stride) {
      firstLayer[i] = values[i];
   }
}

/*
 * Initialise the values of the nodes in the layers.
 * To scale this up, it might be usefull to use templates
 * and higher functions, or just use vectors.
 */
__global__
void layerInit(const unsigned int firstNodes,
               const unsigned int secondNodes,
               const unsigned int resultNodes,
               float *firstLayer,
               float *secondLayer,
               float *resultLayer) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (unsigned int i = index; i < firstNodes; i += stride) {
      firstLayer[i] = 0.0f;
   }
   for (unsigned int i = index; i < secondNodes; i += stride) {
      secondLayer[i] = 0.0f;
   }
   for (unsigned int i = index; i < resultNodes; i += stride) {
      resultLayer[i] = 0.0f;
   }
}

/*
 * Fill the array weights with 'random' numbers.
 * Do note that this is NOT a weightLayer, those will be filled using
 * the values in this array.
 */
__global__
void fillWeights(float *weights, unsigned long int seed, const unsigned int amountWeights) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (unsigned int i = index; i < amountWeights; i += stride) {
      curandState state;
      curand_init(seed, i, 0, &state);
      weights[i] = curand_uniform(&state);
   }
}

/*
 * Given two arrays A = {a,b,c} and Z = {x,y,z}, perform
 * R = A*Z in a manner which gives R = {a*x,b*y,c*z}.
 */
__global__
void multiply(const unsigned int n,
              float *first,
              float *second,
              float *results) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (unsigned int i = index; i < n; i += stride) {
      results[i] = first[i] * second[i];
   }
}

/*
 * allocateStuff allocates stuff.
 * It calls cudaMallocManaged (malloc for cuda) on all necessary
 * arrays and vectors of arrays. This ensures these arrays are
 * in the gpu memory and can therefore be used by cuda.
 */
void allocateStuff(const unsigned int firstNodes,
                   const unsigned int secondNodes,
                   const unsigned int resultNodes,
                   const unsigned int amountWeights,
                   float *&firstLayer,
                   float *&secondLayer,
                   float *&resultLayer,
                   float *&weights,
                   std::vector<float*> &firstWeightLayer,
                   std::vector<float*> &secondWeightLayer) {
   cudaMallocManaged(&firstLayer,  firstNodes  * sizeof(float));
   cudaMallocManaged(&secondLayer, secondNodes * sizeof(float));
   cudaMallocManaged(&resultLayer, resultNodes * sizeof(float));
   cudaMallocManaged(&weights,   amountWeights * sizeof(float));
   
   for (auto& nodeLayer : firstWeightLayer) {
      cudaMallocManaged(&nodeLayer, secondNodes * sizeof(float));
   }
   for (auto& nodeLayer : secondWeightLayer) {
      cudaMallocManaged(&nodeLayer, resultNodes * sizeof(float));
   }
}

/*
 * freeStuff() is a series of three functions which assure that any
 * array or vector of arrays fed to it is freed from the memory.
 */

template<typename T, typename... Args>
void freeStuff(T *t) {
   cudaFree(t);
}

template<typename T, typename... Args>
void freeStuff(T *t, Args... args) {
   freeStuff(t);
   freeStuff(args...);
}

void freeStuff(std::vector<float*> &vec) {
   for (auto& v : vec) {
      freeStuff(v);
   }
}

void weightLayerInit(unsigned int &index,
                     float *&weights,
                     const unsigned int layerLength,
                     std::vector<float *> &vec) {
   //Maybe do something with splicing of weights, so index can be 0 and this
   //function can be global. Or find a way to do that without index being 0.
   //Also remember that vec is a weightLayer, containing X arrays of length
   //layerLength (where X is the amount of nodes in the next layer).
   ;
}

void weightLayerInit(unsigned int &index,
                     float *&weights,
                     const std::vector<unsigned int> &layerLength,
                     std::vector<std::vector<float *>> &vecs) {
   assert(layerLength.size() == vecs.size() || "layerLength and vecs don't have the same size!");
   for (unsigned int i = 0; i < vecs.size(); i++) {
      weightLayerInit(index, weights, layerLength[i], vecs[i]);
      index += layerLength[i];
   }
}

void forward() {

}

int main () {
   unsigned int firstNodes = 5, secondNodes = 3, resultNodes = 1;
   const unsigned int amountWeights = (firstNodes + resultNodes) * secondNodes;
   float *firstLayer;
   float *secondLayer;
   float *resultLayer;
   float *weights;
   //For every node it goes to, from every node it came from
   //This way we can multiply the from nodes with the weights easily
   std::vector<float*> firstWeightLayer(secondNodes, new float[firstNodes]);
   std::vector<float*> secondWeightLayer(resultNodes, new float[secondNodes]);
   
   //Put all the necessary stuff in the gpu memory
   allocateStuff(firstNodes, secondNodes, resultNodes, amountWeights,
                 firstLayer, secondLayer, resultLayer,
                 weights, firstWeightLayer,secondWeightLayer);
   
   //Initialise all the layers to 0. Should not be necessary, still doing it for now
   layerInit<<<1,256>>>(firstNodes,
                        secondNodes,
                        resultNodes,
                        firstLayer,
                        secondLayer,
                        resultLayer);
   
   //Fill an array with 'random' weights. This will be used to initialise the weightLayers
   unsigned long int seed = 12345;
   fillWeights<<<1,256>>>(weights, seed, amountWeights);
   
   unsigned int globalIndex = 0;
   const std::vector<unsigned int> layerLengths = {firstNodes, secondNodes};
   std::vector<std::vector<float *>> weightLayers = {firstWeightLayer, secondWeightLayer};
   weightLayerInit(globalIndex, weights, layerLengths,weightLayers);
   
   //multiply<<<1,256>>>(n, first, second, results);
   
   cudaDeviceSynchronize();
   /*
   float maxError = 0.0f;
   for (unsigned int i = 0; i < n; i++) {
      maxError = fmax(maxError, fabs(results[i]-6.0f));
   }
   std::cout << "Max error: " << maxError << std::endl;
   */
   freeStuff(firstLayer, secondLayer, resultLayer);
   freeStuff(firstWeightLayer);
   freeStuff(secondWeightLayer);
   
   cudaCheckErrors("Hi!");
   //Necessary to be able to use nvprof
   cudaProfilerStop();
   return 0;
}
