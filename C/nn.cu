#include "reader.c"
#include <math.h>


float sigmoid(float x)
{
   return 1/(1+exp(-x));
}

float dsigmoid(float x)
{
   return x*(1-x);
}

void err(cudaError_t returnValue)
{
   if (returnValue != cudaSuccess)
   {
      fprintf(stderr, "CUDA Failure: %s\n", cudaGetErrorString(returnValue));
      exit(EXIT_FAILURE);
   }
}

__global__ void forwardPass1(float* in, float* syn1, float* layer1)
{
   int l = blockDim.x*blockIdx.x + threadIdx.x;
   int X = 28*28;
   int Y = 100;

#pragma unroll
   for (int j=0; j < X; ++j)
      layer1[l] += in[j] * syn1[j*Y + l];

   layer1[l] = 1.0/(1.0 + exp(layer1[l]));
}

__global__ void forwardPass2(float* layer1, float* syn2, float* out)
{
   int l = blockDim.x*blockIdx.x + threadIdx.x;
   int Y = 100;
   int Z = 10;

#pragma unroll
   for (int j=0; j < Y; ++j)
      out[l] += layer1[j] * syn2[j*Z + l];

   out[l] = 1.0/(1.0 + exp(out[l]));
}

__global__ void backProp2(float* layer1, float* dsyn2, float* label, float* out)
{
   int j = blockDim.x*blockIdx.x + threadIdx.x;
   int k = blockDim.y*blockIdx.y + threadIdx.y;
   float delta = (label[k] - out[k]) * (out[k]*(1.0-out[k]));
   dsyn2[j*10 + k] += delta * layer1[j] / (60000.0/10.0);
}

__global__ void backProp1(float* in, float* dsyn1, float* layer1, float* syn2, float* label, float* out)
{
   int j = blockDim.x*blockIdx.x + threadIdx.x;
   int k = blockDim.y*blockIdx.y + threadIdx.y;
   float error = 0.0;

#pragma unroll
   for (int l=0; l < 10; ++l)
      error += (label[l] - out[l]) * syn2[k*10 + l];
   float delta = error * (layer1[k]*(1-layer1[k]));
   dsyn1[j*100 + k] += delta * in[j] / (60000.0/10.0);
}

__global__ void applyChanges(float* syn, float* dsyn, int dim, float alpha)
{
   int j = blockDim.x*blockIdx.x + threadIdx.x;
   int k = blockDim.y*blockIdx.y + threadIdx.y;
   syn[j*dim + k] += dsyn[j*dim + k] * alpha;
}

//__global__ void test(float* in, float* syn1, float* layer1, float* syn2, float* out, float* error,
//                     int X, int Y, int Z)
//{
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

   float input[Data.height][Data.width];                                  //input layer
   float* d_in; err(cudaMalloc((void**)&d_in, Data.height*Data.width*sizeof(float)));
   float weights1[Data.height][Data.width][100];   //input to middle layer weights
   float* d_syn1; err(cudaMalloc((void**)&d_syn1, Data.height*Data.width*100*sizeof(float)));
   float dweights1[Data.height][Data.width][100];  //input to middle layer weights delta
   float* d_dsyn1; err(cudaMalloc((void**)&d_dsyn1, Data.height*Data.width*100*sizeof(float)));
   float layer1[100];                              //Middle layer
   float* d_layer1; err(cudaMalloc((void**)&d_layer1, 100*sizeof(float)));
   float weights2[100][10];                        //middle to output layer weights
   float* d_syn2; err(cudaMalloc((void**)&d_syn2, 100*10*sizeof(float)));
   float dweights2[100][10];                       //middle to output layer weights delta
   float* d_dsyn2; err(cudaMalloc((void**)&d_dsyn2, 100*10*sizeof(float)));
   float outs[10];                                 //Output layer
   float* d_out; err(cudaMalloc((void**)&d_out, 10*sizeof(float)));
   float* d_label; err(cudaMalloc((void**)&d_label, 10*sizeof(float)));
   float alpha = atof(argv[6]);

   //Initialize weights to random values
   printf("randomizing initial weights\n");
   srand(112992); //make the random values the same each time
   for (int i=0; i < Data.height; ++i)
   {
      for (int j=0; j < Data.width; ++j)
      {
         for (int k=0; k < 100; ++k)
         {
            weights1[i][j][k] = (float)rand()/(RAND_MAX/2.0) - 1.0;
            dweights1[i][j][k] = 0;
         }
      }
   }
   for (int i=0; i<100; ++i)
   {
      for (int j=0; j < 10; ++j)
      {
         weights2[i][j] = (float)rand()/(RAND_MAX/2.0) - 1.0;
         //printf("%f, ", weights2[i][j]);
         dweights2[i][j] = 0;
      }
   }

   err(cudaMemcpy(d_syn1, weights1, Data.height*Data.width*100, cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1, dweights1, Data.height*Data.width*100, cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn2, weights2, 10*100, cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn2, dweights2, 10*100, cudaMemcpyHostToDevice));

   //train
   printf("training\n");
   int iterations = atoi(argv[5]);
   for (int iter=0; iter<iterations; ++iter)
   {
      for (int i=0; i < Data.count; ++i)
      {
         //reset layer states
         for (int j=0; j < Data.height; ++j)
         {
            for (int k=0; k < Data.width; ++k)
            {
               input[j][k] = Data.Image[i][j][k];
            }
         }
         for (int j=0; j < 100; ++j)
            layer1[j] = 0.0;
         for (int j=0; j < 10; ++j)
            outs[j] = 0.0;
         err(cudaMemcpy(d_in, input, Data.height*Data.width, cudaMemcpyHostToDevice));
         err(cudaMemcpy(d_layer1, layer1, 100, cudaMemcpyHostToDevice));
         err(cudaMemcpy(d_out, outs, 10, cudaMemcpyHostToDevice));

      // Forward pass
         //input to middle layer
         forwardPass1<<<1, 100>>>(d_in, d_syn1, d_layer1);
         //middle to output layer
         forwardPass2<<<1, 10>>>(d_layer1, d_syn2, d_out);
         
      //Back Propagation
      //   error[k] = labels[i][k] - outs[k]
      //   delta[k] = error * dsigmoid(outs[k])
      //   weights2[j][k] += delta[k] * layer1[j]
         err(cudaMemcpy(d_label, Data.Label[i], 10, cudaMemcpyHostToDevice));
         //output to middle
         backProp2<<<dim3(10,1), dim3(10,10)>>>(d_layer1, d_dsyn2, d_label, d_out);
         //middle to input
         backProp1<<<dim3(28, 10), dim3(28, 10)>>>(d_in, d_dsyn1, d_layer1, d_syn2, d_label, d_out);
      }

      //adjust synapse weights
      applyChanges<<<dim3(10,1), dim3(10,10)>>>(d_syn2, d_dsyn2, 10, alpha);
      applyChanges<<<dim3(28,10), dim3(28,10)>>>(d_syn1, d_dsyn1, 100, alpha);

      for (int i=0; i < Data.height; ++i)
      {
         for (int j=0; j < Data.width; ++j)
         {
            for (int k=0; k < 100; ++k)
            {
               dweights1[i][j][k] = 0.0;
            }
         }
      }
      //printf("\n");
      for (int i=0; i < 100; ++i)
      {
         for (int j=0; j < 10; ++j)
         {
            dweights2[i][j] = 0.0;
         }
      }
      err(cudaMemcpy(d_dsyn1, dweights1, 28*28*100, cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn2, dweights2, 100*10, cudaMemcpyHostToDevice));
      printf("%d\n", iter);
   }
   
   //copy synapse weights back to host memory
   err(cudaMemcpy(weights1, d_syn1, 28*28*100, cudaMemcpyDeviceToHost));
   err(cudaMemcpy(weights2, d_syn2, 100*10, cudaMemcpyDeviceToHost));

   //test
   printf("testing\n");
   float error = 0.0;
   //for (int i=0; i < Test.height; ++i)
   //{
   //   for (int j=0; j < Test.width; ++j)
   //   {
   //      printf("%f ", Test.Image[0][i][j]);
   //   }
   //   printf("\n");
   //}
   for (int i=0; i < Test.count; ++i)
   {

      //reset layer states
      for (int j=0; j < Test.height; ++j)
      {
         for (int k=0; k < Test.width; ++k)
         {
            input[j][k] = Test.Image[i][j][k];
         }
      }
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
               layer1[l] += input[j][k] * weights1[j][k][l];
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
            outs[k] += layer1[j] * weights2[j][k];
         }
      }
      for (int j=0; j < 10; ++j)
      {
         outs[j] = sigmoid(outs[j]);
         printf("%f ", outs[j]);
      }
      printf("\n");

      //sum up error
      for (int j=0; j < 10; ++j)
      {
         printf("%f ", Test.Label[i][j]);
         error += fabs(Test.Label[i][j] - outs[j])/10.0;
      }
      printf("\n");
   }
   //printf("Error: %f\n", error);
   error /= Test.count;
   printf("Error: %f percent\n", error*100.0);

   //clean up CUDA arrays on GPU

   err(cudaFree(d_in));
   err(cudaFree(d_syn1));
   err(cudaFree(d_dsyn1));
   err(cudaFree(d_layer1));
   err(cudaFree(d_syn2));
   err(cudaFree(d_dsyn2));
   err(cudaFree(d_out));
   err(cudaFree(d_label));

   //clean up data arrays
   for (int i=0; i<Data.count; ++i)
   {
      for (int j=0; j<Data.height; ++j)
      {
         free(Data.Image[i][j]);
      }
      free(Data.Image[i]);
      free(Data.Label[i]);
   }
   free(Data.Image);
   free(Data.Label);
   for (int i=0; i<Test.count; ++i)
   {
      for (int j=0; j<Test.height; ++j)
      {
         free(Test.Image[i][j]);
      }
      free(Test.Image[i]);
      free(Test.Label[i]);
   }
   free(Test.Image);
   free(Test.Label);
   return 0;
}
