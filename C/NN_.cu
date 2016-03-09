#include "reader_.c"
#include <math.h>
#include <time.h>


float sigmoid(float x)
{
   return 1/(1+exp(-x));
}

float dsigmoid(float x)
{
   return x*(1-x);
}

void err(cudaError_t returnVal)
{
   if (returnVal != cudaSuccess)
   {
      fprintf(stderr, "CUDA Failure: %s\n", cudaGetErrorString(returnVal));
      exit(EXIT_FAILURE);
   }
}

__global__ void train(const float* in, const float* label, const float* syn1, const float* syn2, float* dsyn1, float* dsyn2, const float alpha)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x;

   if (i < 60000)
   {
      float layer1[100] = {0.0};            //Middle layer
      float outs[10] = {0.0};               //Output layer

      // Forward pass
      //input to middle layer
#pragma unroll 28
      for (int j=0; j < 28; ++j)
      {
#pragma unroll 28
         for (int k=0; k < 28; ++k)
         {
#pragma unroll 100
            for (int l=0; l < 100; ++l)
            {
               layer1[l] += in[i*28*28 + j*28 + k] * syn1[j*28*100 + k*100 + l];
            }
         }
      }
#pragma unroll 100
      for (int j=0; j < 100; ++j)
         layer1[j] = 1/(1+exp(-layer1[j]));
      
      //middle to output layer
#pragma unroll 100
      for (int j=0; j < 100; ++j)
      {
#pragma unroll 10
         for (int k=0; k < 10; ++k)
         {
            outs[k] += layer1[j] * syn2[j*10 + k];
         }
      }
#pragma unroll 10
      for (int j=0; j < 10; ++j)
      {
         outs[j] = 1/(1+exp(-outs[j]));
      }

      //Back Propagation
      //   error[k] = labels[i][k] - outs[k]
      //   delta[k] = error * dsigmoid(outs[k])
      //   weights2[j][k] += delta[k] * layer1[j]
      //output to middle
      float delta2[10] = {0};
      //__shared__ float psyn2[10*100];
#pragma unroll 100
      for (int j=0; j < 100; ++j)
      {
#pragma unroll 10
         for (int k=0; k < 10; ++k)
         {
            delta2[k] = (label[i*10 + k] - outs[k]) * outs[k]*(1.0-outs[k]);
            atomicAdd(&dsyn2[j*10 + k], delta2[k] * alpha * layer1[j] / (gridDim.x*blockDim.x));
         }
      }
      //middle to input
      float delta1[100] = {0.0};
      float error1[100] = {0.0};
      //__shared__ float psyn1[28*28*100];
#pragma unroll 28
      for (int h=0; h < 28; ++h)
      {
#pragma unroll 28
         for (int j=0; j < 28; ++j)
         {
#pragma unroll 100
            for (int k=0; k < 100; ++k)
            {
               error1[k] = 0.0;
#pragma unroll 10
               for (int l=0; l < 10; ++l)
                  error1[k] += delta2[l] * syn2[k*10 * l];
               delta1[k] = error1[k] * layer1[k]*(1.0-layer1[k]);
               atomicAdd(&dsyn1[h*28*100 + j*100 + k], alpha * (delta1[k] * in[i*28*28 + h*28 + j] / (gridDim.x*blockDim.x)));
            }
         }
      }
      //__syncthreads();

      //if (threadIdx.x == 0)
      //{
      //   for (int j=0; j < 100; ++j)
      //   {
      //      for (int k=0; k < 10; ++k)
      //      {
      //         atomicAdd(&dsyn2[j*10 + k], psyn2[j*10 + k]);
      //      }
      //   }
      //   for (int h=0; h < 28; ++h)
      //   {
      //      for (int j=0; j < 28; ++j)
      //      {
      //         for (int k=0; k < 100; ++k)
      //         {
      //            atomicAdd(&dsyn1[h*28*100 + j*100 + k], psyn1[h*28*100 + j*100 + k]);
      //         }
      //      }
      //   }
      //}
   }
}

//__global__ void apply(float* out, float* in, int n)
//{
//   int i = blockDim.x*blockIdx.x + threadIdx.x;
//   if (i < n)
//   {
//      atomicAdd(&out[i], in[i]);
//      in[i] = 0.0;
//   }
//}

int main(int argc, char** argv)
{
   if (argc != 7)
   {
      printf("usage: run trainingImages trainingLabels testImages testLabels iterations alpha\n");
      return 2;
   }
   struct data Data = read(argv[1], argv[2]);      //training data
   struct data Test = read(argv[3], argv[4]);      //test data

   float weights1[28*28*100];   //input to middle layer weights
   //float dweights1[28*28*100];  //input to middle layer weights
   float layer1[100];                              //Middle layer
   float weights2[100*10];                        //middle to output layer weights
   //float dweights2[100*10];                       //middle to output layer weights
   float outs[10];                                 //Output layer
   float alpha = atof(argv[6]);

   float* d_in;    err(cudaMalloc((void**)&d_in,    28*28*60000*sizeof(float)));
   float* d_label; err(cudaMalloc((void**)&d_label, 60000*10*sizeof(float)));
   float* d_syn1;  err(cudaMalloc((void**)&d_syn1,  28*28*100*sizeof(float)));
   float* d_dsyn1; err(cudaMalloc((void**)&d_dsyn1, 28*28*100*sizeof(float)));
   float* d_syn2;  err(cudaMalloc((void**)&d_syn2,  100*10*sizeof(float)));
   float* d_dsyn2; err(cudaMalloc((void**)&d_dsyn2, 100*10*sizeof(float)));

   //Initialize weights to random values
   //printf("randomizing initial weights\n");
   srand(1); //make the random values the same each time
   for (int i=0; i < 28; ++i)
   {
      for (int j=0; j < 28; ++j)
      {
         for (int k=0; k < 100; ++k)
         {
            weights1[i*28*100 + j*100 + k] = (float)rand()/(RAND_MAX/2.0) - 1.0;
            //dweights1[i*28*100 + j*100 + k] = 0.0;
         }
      }
   }
   for (int i=0; i<100; ++i)
   {
      for (int j=0; j < 10; ++j)
      {
         weights2[i*10 + j] = (float)rand()/(RAND_MAX/2.0) - 1.0;
         //dweights2[i*10 + j] = 0.0;
      }
   }

   clock_t start = clock();

   err(cudaMemcpy(d_in, Data.Image, 28*28*60000*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_label, Data.Label, 10*60000*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn1, weights1, 28*28*100*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1, d_syn1, 28*28*100*sizeof(float), cudaMemcpyDeviceToDevice));
   err(cudaMemcpy(d_syn2, weights2, 10*100*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn2, d_syn2, 10*100*sizeof(float), cudaMemcpyDeviceToDevice));

   //float* testI = (float*)malloc(28*28*60000*sizeof(float));
   //float* testL = (float*)malloc(10*60000*sizeof(float));
   //bool potato = false;
   //err(cudaMemcpy(testI, d_in, 28*28*60000*sizeof(float), cudaMemcpyDeviceToHost));
   //err(cudaMemcpy(testL, d_label, 10*60000*sizeof(float), cudaMemcpyDeviceToHost));
   //for (int i=0; i < 28*28*60000; ++i)
   //{
   //   if (testI[i] > 1.0 || testI[i] < 0.0)
   //      potato = true;
   //}
   //if (potato)
   //   printf("error in the input data\n");
   //potato = false;
   //for (int i=0; i < 10*60000; ++i)
   //{
   //   if (testL[i] > 1.0 || testL[i] < 0.0)
   //      potato = true;
   //}
   //if (potato)
   //   printf("error in the label data\n");
   //potato = false;

   //train
   //printf("training\n");
   int iterations = atoi(argv[5]);
   float al = 1.0;
   for (int iter=0; iter<iterations; ++iter)
   {
      if (iter < 8)
         al = alpha*20.0;
      else if (iter < 16)
         al = alpha*10.0;
      else if (iter < 32)
         al = alpha*5.0;
      else if (iter < 64)
         al = alpha*2.0;
      else
         al = alpha;
      train<<<48,   125>>>(&d_in[6000*(iter%10)], &d_label[6000*(iter%10)], d_syn1, d_syn2, d_dsyn1, d_dsyn2, al);
      err(cudaMemcpy(d_syn1, d_dsyn1, sizeof(float)*28*28*100, cudaMemcpyDeviceToDevice));
      err(cudaMemcpy(d_syn2, d_dsyn2, sizeof(float)*100*10,    cudaMemcpyDeviceToDevice));

      //err(cudaMemcpy(testI, d_in, 28*28*60000*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(testL, d_label, 10*60000*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights1, d_syn1, 28*28*100*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights2, d_syn2, 10*100*sizeof(float), cudaMemcpyDeviceToHost));
      //for (int i=0; i < 28*28*60000; ++i)
      //{
      //   if (testI[i] > 1.0 || testI[i] < 0.0 || isnan(testI[i]))
      //      potato = true;
      //}
      //if (potato)
      //   printf("error in the input data\n");
      //potato = false;
      //for (int i=0; i < 10*60000; ++i)
      //{
      //   if (testL[i] > 1.0 || testL[i] < 0.0 || isnan(testL[i]))
      //      potato = true;
      //}
      //if (potato)
      //   printf("error in the label data\n");
      //potato = false;
      //for (int i=0; i < 28*28*100; ++i)
      //{
      //   if (isnan(weights1[i]))
      //   {
      //      potato = true;
      //      printf("syn1: %d: %f\n", i, weights1[i]);
      //   }
      //}
      //potato = false;
      //for (int i=0; i < 10*100; ++i)
      //{
      //   if (isnan(weights2[i]))
      //   {
      //      potato = true;
      //      printf("syn1: %d: %f\n", i, weights2[i]);
      //   }
      //}
   }
   //free(testI);
   //free(testL);
   err(cudaMemcpy(weights1, d_syn1, sizeof(float)*28*28*100, cudaMemcpyDeviceToHost));
   err(cudaMemcpy(weights2, d_syn2, sizeof(float)*100*10,    cudaMemcpyDeviceToHost));

   clock_t diff = clock() - start;
   diff = diff*1000/CLOCKS_PER_SEC;
   printf("computation time: %ld.%ld\n", diff/1000, diff%1000);

   //test
   //printf("testing\n");
   float error = 0.0;
   //for (int i=0; i < 100; ++i)
   //{
   //   for (int j=0; j < 10; ++j)
   //   {
   //      printf("%f ", weights2[i*10 + j]);
   //   }
   //   printf("\n");
   //}
   for (int i=0; i < Test.count; ++i)
   {

      //reset layer states
      for (int j=0; j < 100; ++j)
         layer1[j] = 0.0;
      for (int j=0; j < 10; ++j)
         outs[j] = 0.0;

      // Forward pass
      //input to middle layer
      for (int j=0; j < Test.height; ++j)
      {
         for (int k=0; k < Test.width; ++k)
         {
            for (int l=0; l < 100; ++l)
            {
               layer1[l] += Test.Image[i*28*28 + j*28 + k] * weights1[j*28*100 + k*100 + l];
            }
         }
      }
      for (int j=0; j < 100; ++j)
         layer1[j] = sigmoid(layer1[j]);

      //middle to output layer
      for (int j=0; j < 100; ++j)
      {
         for (int k=0; k < 10; ++k)
         {
            outs[k] += layer1[j] * weights2[j*10 + k];
         }
      }
      for (int j=0; j < 10; ++j)
      {
         outs[j] = sigmoid(outs[j]);
         //printf("%f ", outs[j]);
      }
      //printf("\n");

      //sum up error
      for (int j=0; j < 10; ++j)
      {
         //printf("%f ", Test.Label[i*10 + j]);
         error += fabs(Test.Label[i*10 + j] - outs[j])/10.0;
      }
      //printf("\n");
   }
   //printf("Error: %f\n", error);
   error /= Test.count;
   printf("Error: %f %%\n", error*100.0);

   //clean up data arrays
   //for (int i=0; i<60000; ++i)
   //{
   //   for (int j=0; j<28; ++j)
   //   {
   //      free(Data.Image[i][j]);
   //   }
   //   free(Data.Image[i]);
   //   free(Data.Label[i]);
   //}
   free(Data.Image);
   free(Data.Label);
   //for (int i=0; i<Test.count; ++i)
   //{
   //   for (int j=0; j<Test.height; ++j)
   //   {
   //      free(Test.Image[i][j]);
   //   }
   //   free(Test.Image[i]);
   //   free(Test.Label[i]);
   //}
   free(Test.Image);
   free(Test.Label);

   err(cudaFree(d_in));
   err(cudaFree(d_label));
   err(cudaFree(d_syn1));
   err(cudaFree(d_syn2));
   err(cudaFree(d_dsyn1));
   err(cudaFree(d_dsyn2));

   return EXIT_SUCCESS;
}
