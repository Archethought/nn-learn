#include "reader_.c"
#include <math.h>
#include <time.h>



void err(cudaError_t returnVal)
{
   if (returnVal != cudaSuccess)
   {
      fprintf(stderr, "CUDA Failure: %s\n", cudaGetErrorString(returnVal));
      exit(EXIT_FAILURE);
   }
}

__device__ float sigmoid(float x)
{
   return 1.0/(1.0+expf(-x));
}
__device__ float dsigmoid(float x)
{
   return x*(1.0-x);
}

__global__ void Fprop1(const float* in, const float* syn1, float* layer1)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //Data.count

   float x = 0.0;
   for(int k=0; k < 28*28; ++k)
      x += in[28*28*j + k] * syn1[256*k + i];
   layer1[256*j + i] = tanh(x);
}
__global__ void Fprop2(const float* layer1, const float* syn2, float* out)
{
   int i = threadIdx.x; //10
   int j = blockIdx.x;  //Data.count

   float x = 0.0;
   for(int k=0; k < 256; ++k)
      x += layer1[256*j + k] * syn2[10*k + i];
   out[10*j + i] = sigmoid(x);
}
__global__ void Dcalc2(float* out, const float* label)
{
   int i = threadIdx.x; //10
   int j = blockIdx.x;  //Data.coun

   float x = label[10*j + i] - out[10*j + i];
   out[10*j + i] = x * powf(9.0, label[10*j + i]) / 9.0;
}
__global__ void Ecalc2(float* out, const float* label)
{
   int i = threadIdx.x; //10
   int j = blockIdx.x;  //Data.coun

   float x = label[10*j + i] - out[10*j + i];
   out[10*j + i] = -x;
}
__global__ void Bprop2(const float* layer1, float* dsyn2, const float* out, const float alpha)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //Data.count

   for(int k=0; k < 10; ++k)
   {
      atomicAdd(&dsyn2[10*i + k], out[10*j + k] * layer1[256*j + i] * alpha);
   }
}
__global__ void Dcalc1(float* layer1, const float* syn2, const float* out)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //Data.count

   float x = 0.0;
   for(int k=0; k < 10; ++k)
   {
      x += out[10*j + k] * syn2[10*i + k];
   }
   //float y = layer1[256*j + i];
   layer1[256*j + i] = x;// * (1-y*y); //dtanh(y)
}
__global__ void Bprop1(const float* in, float* dsyn1, const float* layer1, const float alpha)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //Data.count

   for(int k=0; k < 28*28; ++k)
   {
      atomicAdd(&dsyn1[256*k + i], layer1[256*j + i] * in[28*28*j + k] * alpha);
   }
}


float randGen()
{
   // pseudo-normal distribution using 3 random numbers
   // min/max = +/- 1
   // mean = 0
   // std dev = 0.5
   float r1 = (float)rand()/RAND_MAX*2 - 1;
   float r2 = (float)rand()/RAND_MAX*2 - 1;
   float r3 = (float)rand()/RAND_MAX*2 - 1;
   return (r1 + r2 + r3)/3.0;
}

int main(int argc, char** argv)
{
   if (argc != 7)
   {
      printf("usage: run trainingImages trainingLabels testImages testLabels iterations alpha\n");
      return 2;
   }
   struct data Data = read_in(argv[1], argv[2]);      //training data
   struct data Test = read_in(argv[3], argv[4]);      //test data

   //neuron layers
   float* d_in;     err(cudaMalloc((void**)&d_in,     Data.count*28*28*sizeof(float)));
   float* d_label;  err(cudaMalloc((void**)&d_label,  Data.count*10*sizeof(float)));
   float* d_layer1; err(cudaMalloc((void**)&d_layer1, Data.count*256*sizeof(float)));
   float* d_out;    err(cudaMalloc((void**)&d_out,    Data.count*10*sizeof(float)));

   //synapses between layers
   //   and also a seperate set to hold the updates made during training
   float* d_syn1;   err(cudaMalloc((void**)&d_syn1,   28*28*256*sizeof(float)));
   float* d_dsyn1;  err(cudaMalloc((void**)&d_dsyn1,  28*28*256*sizeof(float)));
   float* d_syn2;   err(cudaMalloc((void**)&d_syn2,   256*10*sizeof(float)));
   float* d_dsyn2;  err(cudaMalloc((void**)&d_dsyn2,  256*10*sizeof(float)));
   float* weights1 = (float*)malloc(sizeof(float)*28*28*256);
   float* weights2 = (float*)malloc(sizeof(float)*256*10);

   // randomize initial synapse weights
   srand(112992);
   for(int i=0; i < 28*28*256; ++i)
   {
      float r = 2.0/sqrt(28*28);
      weights1[i] = randGen()*r;
   }
   for(int i=0; i < 256*10; ++i)
   {
      float r = 2.0/sqrt(256);
      weights2[i] = randGen()*r;
   }

   clock_t start = clock();

   err(cudaMemcpy(d_in, Data.Image, 28*28*Data.count*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_label, Data.Label, 10*Data.count*sizeof(float), cudaMemcpyHostToDevice));
   //err(cudaMemset(d_layer1,   0.0,          256*sizeof(float)));
   //err(cudaMemset(d_out,      0.0,           10*sizeof(float)));
   err(cudaMemcpy(d_syn1,  weights1,  28*28*256*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1, weights1,  28*28*256*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn2,  weights2,     256*10*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn2, weights2,     256*10*sizeof(float), cudaMemcpyHostToDevice));

   float* layer1 = (float*)malloc(256*Data.count*sizeof(float));

//////// Training /////////
   int iterations = atoi(argv[5]);
   float alpha = atof(argv[6]);
   //printf("%f\n", alpha);
   for (int iter=0; iter<iterations; ++iter)
   {
      err(cudaMemset(d_layer1, 0.0, 256*Data.count*sizeof(float)));
      err(cudaMemset(d_out,    0.0,  10*Data.count*sizeof(float)));

      err(cudaMemcpy(layer1, d_layer1, 256*Data.count*sizeof(float), cudaMemcpyDeviceToHost));
      for(int i=0; i < 10; ++i)
         printf("%f\t", layer1[i]);
      printf("\n");

   //Forward Propagation
      Fprop1<<<Data.count, 256>>>(d_in, d_syn1, d_layer1);
      Fprop2<<<Data.count, 10>>>(d_layer1, d_syn2, d_out);

      err(cudaMemcpy(layer1, d_layer1, 256*Data.count*sizeof(float), cudaMemcpyDeviceToHost));
      for(int i=0; i < 10; ++i)
         printf("%f\t", layer1[i]);
      printf("\n");

   //Backpropagation
      Dcalc2<<<Data.count, 10>>>(d_out, d_label);
      Bprop2<<<Data.count, 256>>>(d_layer1, d_dsyn2, d_out, alpha/Data.count);
      Dcalc1<<<Data.count, 256>>>(d_layer1, d_syn2, d_out);

      err(cudaMemcpy(layer1, d_layer1, 256*Data.count*sizeof(float), cudaMemcpyDeviceToHost));
      for(int i=0; i < 10; ++i)
         printf("%f\t", layer1[i]);
      printf("\n");

      err(cudaMemcpy(weights1, d_dsyn1, 256*28*28*sizeof(float), cudaMemcpyDeviceToHost));
      for(int i=0; i < 10; ++i)
         printf("%f\t", weights1[i]);
      printf("\n");

      Bprop1<<<Data.count, 256>>>(d_in, d_dsyn1, d_layer1, alpha/Data.count);

      err(cudaMemcpy(layer1, d_layer1, 256*Data.count*sizeof(float), cudaMemcpyDeviceToHost));
      for(int i=0; i < 10; ++i)
         printf("%f\t", layer1[i]);
      printf("\n");

      err(cudaMemcpy(weights1, d_dsyn1, 256*28*28*sizeof(float), cudaMemcpyDeviceToHost));
      for(int i=0; i < 10; ++i)
         printf("%f\t", weights1[i]);
      printf("\n");

      err(cudaMemcpy(d_syn1, d_dsyn1, sizeof(float)*28*28*256, cudaMemcpyDeviceToDevice));
      err(cudaMemcpy(d_syn2, d_dsyn2, sizeof(float)*256*10,    cudaMemcpyDeviceToDevice));
   }

   err(cudaMemcpy(weights1, d_syn1, sizeof(float)*28*28*256, cudaMemcpyDeviceToHost));
   err(cudaMemcpy(weights2, d_syn2, sizeof(float)*256*10,    cudaMemcpyDeviceToHost));

   clock_t diff = clock() - start;
   diff = diff*1000/CLOCKS_PER_SEC;
   printf("computation time: %ld.%ld\n", diff/1000, diff%1000);

   err(cudaFree(d_in));
   err(cudaFree(d_label));
   err(cudaFree(d_layer1));
   err(cudaFree(d_out));

//////// Testing /////////
   float error = 0.0;
   
   float* out = (float*)malloc(10*Test.count*sizeof(float));
   err(cudaMalloc((void**)&d_in,     Test.count*28*28*sizeof(float)));
   err(cudaMalloc((void**)&d_label,  Test.count*10*sizeof(float)));
   err(cudaMalloc((void**)&d_layer1, Test.count*256*sizeof(float)));
   err(cudaMalloc((void**)&d_out,    Test.count*10*sizeof(float)));
   cudaMemset(d_layer1, 0.0, 256*Test.count*sizeof(float));
   cudaMemset(d_out,    0.0,  10*Test.count*sizeof(float));
   err(cudaMemcpy(d_in, Test.Image, 28*28*Test.count*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_label, Test.Label, 10*Test.count*sizeof(float), cudaMemcpyHostToDevice));

   //Forward Propagation
   Fprop1<<<Test.count, 256>>>(d_in, d_syn1, d_layer1);
   Fprop2<<<Test.count, 10>>>(d_layer1, d_syn2, d_out);

   Ecalc2<<<Test.count, 10>>>(d_out, d_label);
   err(cudaMemcpy(out, d_out, 10*Test.count*sizeof(float), cudaMemcpyDeviceToHost));

   float max = 0.0;
   for(int i=0; i < 10*Test.count; ++i)
   {
      if (i < 10)
         printf("%f\t", out[i]);
      else if (i == 10)
         printf("\n\n");
      if (abs(out[i]) > max)
         max = abs(out[i]);
      error += abs(out[i]);
   }
   error /= 10*Test.count;
   for(int i=0; i < 10; ++i)
      printf("%f\t", weights2[i]);
   printf("\n\n");
   for(int i=0; i < 10; ++i)
      printf("%f\t", weights1[i]);
   printf("\n\n");

   printf("Max: %f\n", max);
   printf("Error: %f%%\n", error * 100.0);

   free(out);
   free(weights1);
   free(weights2);
   free(Data.Image);
   free(Data.Label);
   free(Test.Image);
   free(Test.Label);
   err(cudaFree(d_in));
   err(cudaFree(d_label));
   err(cudaFree(d_layer1));
   err(cudaFree(d_out));
   err(cudaFree(d_syn1));
   err(cudaFree(d_dsyn1));
   err(cudaFree(d_syn2));
   err(cudaFree(d_dsyn2));
   return EXIT_SUCCESS;
}
