#include "reader_ts.c"


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

__global__ void Fprop1(const float* in, const float* syn1, float* layer1)
{
   int i = threadIdx.x;                         //128
   int j = blockDim.y*blockIdx.y + threadIdx.y; //64
   int k = blockIdx.x;                          //Data.count
   atomicAdd(&layer1[128*k + i], in[64*k + j] * syn1[j*128 + i]);
}
__global__ void FpropH(float* layer1, const float* synH, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x;
   int j = blockDim.y*blockIdx.y + threadIdx.y;
   atomicAdd(&layer1[128*offset + i], layer1[128*(offset-1) + j] * synH[j*128 + i]);
   //__syncthreads();
   //if (i == 0)
   //   layerH[j] = layer1[j];
}
__global__ void Fprop2(const float* layer1, const float* syn2, float* out)
{
   int i = threadIdx.x; //4
   int j = blockIdx.x;  //Data.count
   int k = threadIdx.y; //128
   atomicAdd(&out[4*j + i], layer1[128*j + k] * syn2[k*4 + i]);
}
__global__ void Dcalc2(float* out, const float* label)
{
   int i = threadIdx.x;
   int j = blockDim.y*blockIdx.y + threadIdx.y;

   out[4*j + i] = label[i] - out[4*j + i];
}
__global__ void Bprop2(const float* out, const float* layer1, float* dsyn2, const float alpha)
{
   int i = threadIdx.y;
   int j = threadIdx.x;
   int k = blockIdx.x;

   atomicAdd(&dsyn2[i*4 + j], out[4*k + j] * layer1[128*k + i] * alpha);
}
__global__ void Dcalc1(const float* out, float* dlayer1, const float* syn2)
{
   int i = threadIdx.x;
   int j = blockDim.y*blockIdx.y + threadIdx.y;

#pragma unroll
   for (int k=0; k < 4; ++k)
      dlayer1[j*128 + i] = out[j*128 + k] * syn2[i*4 + k];
}
__global__ void BpropH(const float* layer1, const float* dlayer1, float* dsynH, const float alpha, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x;
   int j = blockDim.y*blockIdx.y + threadIdx.y;

   dsynH[i*128 + j] += dlayer1[offset*128 + i] * layer1[(offset-1)*128 + j] * alpha;
}
__global__ void Bprop1(const float* dlayer1, const float* in, const float* syn1, float* dsyn1, const float alpha)
{
   int i = blockDim.y*blockIdx.y + threadIdx.y;
   int j = threadIdx.x;
   int k = blockIdx.x;

   atomicAdd(&dsyn1[i*128 + j], dlayer1[k*128 + j] * in[k*128 + j] * alpha);
}

int main(int argc, char** argv)
{
   //if (argc != 4)
   //{
   //   printf("usage: run trainingImage iterations alpha\n");
   //   return 2;
   //}
   //printf("%d\n", argc);
   const char* genre[4] = {"classical", "jazz", "metal", "pop"};
   int locations[5] = {0};
   int num_data = argc-7;
   int num_test = num_data/4;
   num_data -= num_test;

   if (num_data == 0 || num_test == 0)
   {  printf("too few data\n"); exit(1); }

   printf("reading data\n");
   struct data* Data = (struct data*)malloc(num_data*sizeof(struct data));
   struct data* Test = (struct data*)malloc(num_test*sizeof(struct data));
   for (int i=1; i < argc-2; ++i)
   {
      for (int j=0; j < 4; ++j)
      {
         if (!strcmp(argv[i], genre[j]))
            locations[j] = i;
      }
   }
   locations[4] = argc-2;
   for (int i=0; i < 5; ++i)
      //printf("%d\n", locations[i]);
   for (int j=0, k=0, l=0; j<4; ++j)
   {
      //int split = locations[j] + 0.7*(locations[j+1] - locations[j]);
      int split = (num_data)/4;
      //printf("%d\n", split);
      for (int i=locations[j]+1; i < locations[j+1]; ++i)
      {
         if (i-locations[j]-1 < split)
         {
            Data[k] = read_this(argv[i], genre[j]); //training data
            ++k;
         }
         else
         {
            Test[l] = read_this(argv[i], genre[j]); //testing data
            ++l;
         }
         //printf("%d, %d, %d:%x, %d:%x\n", i, i<split, k, Data[k-1].Image, l, Test[l-1].Image);
      }
   }

   float weights1[64*128];   //input to middle layer weights
   //float layer1[128] = {0};                              //Middle layer
   float weights2[128*4];                        //middle to output layer weights
   float weightsH[128*128];
   //float outs[4] = {0};                                 //Output layer
   float alpha = atof(argv[argc-1]);

   //float* d_in;     err(cudaMalloc((void**)&d_in,     64*Data.count*sizeof(float)));
   //float* d_label;  err(cudaMalloc((void**)&d_label,  Data.count*4*sizeof(float)));
   //float* d_layer1; err(cudaMalloc((void**)&d_layer1, Data.count*128*sizeof(float)));
   //float* d_dlayer1;err(cudaMalloc((void**)&d_dlayer1,Data.count*128*sizeof(float)));
   //float* d_out;    err(cudaMalloc((void**)&d_out,    Data.count*4*sizeof(float)));
   float* d_syn1;   err(cudaMalloc((void**)&d_syn1,   64*128*sizeof(float)));
   float* d_dsyn1;  err(cudaMalloc((void**)&d_dsyn1,  64*128*sizeof(float)));
   float* d_synH;   err(cudaMalloc((void**)&d_synH,   128*128*sizeof(float)));
   float* d_dsynH;  err(cudaMalloc((void**)&d_dsynH,  128*128*sizeof(float)));
   float* d_syn2;   err(cudaMalloc((void**)&d_syn2,   128*4*sizeof(float)));
   float* d_dsyn2;  err(cudaMalloc((void**)&d_dsyn2,  128*4*sizeof(float)));
   float* d_label;  err(cudaMalloc((void**)&d_label,  4*sizeof(float)));

   //Initialize weights to random values
   //printf("randomizing initial weights\n");
   srand(112992); //make the random values the same each time
   for (int j=0; j < 64; ++j)
   {
      for (int k=0; k < 128; ++k)
      {
         weights1[j*128 + k] = (float)rand()/(RAND_MAX/2.0) - 1.0;
      }
   }
   for (int i=0; i<128; ++i)
   {
      for (int j=0; j < 4; ++j)
      {
         weights2[i*4 + j] = (float)rand()/(RAND_MAX/2.0) - 1.0;
      }
   }

   for (int i=0; i<128; ++i)
   {
      for (int j=0; j < 128; ++j)
      {
         weightsH[i*128 + j] = (float)rand()/(RAND_MAX/2.0) - 1.0;
      }
   }


   //err(cudaMemcpy(d_in, Data.Image, 64*Data.count*sizeof(float), cudaMemcpyHostToDevice));
   //err(cudaMemcpy(d_label, Data.Label, 4*Data.count*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn1, weights1, 64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1, d_syn1, 64*128*sizeof(float), cudaMemcpyDeviceToDevice));
   err(cudaMemcpy(d_syn2, weights2, 4*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn2, d_syn2, 4*128*sizeof(float), cudaMemcpyDeviceToDevice));
   err(cudaMemcpy(d_synH, weightsH, 128*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsynH, d_synH, 128*128*sizeof(float), cudaMemcpyDeviceToDevice));

   //train
   //printf("training\n");
   int iterations = atoi(argv[argc-2]);
   printf("training %d iterations\n", iterations);
   for (int iter=0; iter<iterations; ++iter)
   {
      for (int d=0; d < num_data; ++d)
      {
         float* d_in;     
         float* d_layer1; 
         float* d_dlayer1;
         float* d_out;    
         err(cudaMalloc((void**)&d_in,     64*Data[d].count*sizeof(float)));
         err(cudaMalloc((void**)&d_layer1, Data[d].count*128*sizeof(float)));
         err(cudaMalloc((void**)&d_dlayer1,Data[d].count*128*sizeof(float)));
         err(cudaMalloc((void**)&d_out,    Data[d].count*4*sizeof(float)));

         err(cudaMemcpy(d_in, Data[d].Image, 64*Data[d].count*sizeof(float), cudaMemcpyHostToDevice));
         err(cudaMemcpy(d_label, Data[d].Label, 4*sizeof(float), cudaMemcpyHostToDevice));

         //train<<<48,   125>>>(&d_in[6000*(iter%4)], &d_label[6000*(iter%4)], d_syn1, d_syn2, d_dsyn1, d_dsyn2, alpha);
         Fprop1<<<dim3(Data[d].count,16), dim3(128,4)>>>(d_in, d_syn1, d_layer1);
         for (int i=1; i < Data[d].count; ++i)
            FpropH<<<dim3(16,1), dim3(4,128)>>>(d_layer1, d_synH, i);
         Fprop2<<<dim3(Data[d].count,1), dim3(4,128)>>>(d_layer1, d_syn2, d_out);
         Dcalc2<<<dim3(1,Data[d].count), dim3(4,1)>>>(d_out, d_label);
         Bprop2<<<dim3(Data[d].count,1), dim3(4,128)>>>(d_out, d_layer1, d_dsyn2, alpha);
         Dcalc1<<<dim3(1,Data[d].count), dim3(128,1)>>>(d_out, d_dlayer1, d_syn2);
         for (int i=Data[d].count-1; i > 0; --i)
            BpropH<<<dim3(32,1), dim3(4,128)>>>(d_layer1, d_dlayer1, d_dsynH, alpha, i);
         Bprop1<<<dim3(Data[d].count,16), dim3(128,4)>>>(d_dlayer1, d_in, d_syn1, d_dsyn1, alpha);
         err(cudaMemcpy(d_syn1, d_dsyn1, sizeof(float)*64*128, cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_syn2, d_dsyn2, sizeof(float)*128*4,    cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_synH, d_dsynH, sizeof(float)*128*128,   cudaMemcpyDeviceToDevice));

         err(cudaFree(d_in));
         err(cudaFree(d_layer1));
         err(cudaFree(d_dlayer1));
         err(cudaFree(d_out));
      }
   }
   //free(testI);
   //free(testL);
   err(cudaMemcpy(weights1, d_syn1, sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
   err(cudaMemcpy(weights2, d_syn2, sizeof(float)*128*4,  cudaMemcpyDeviceToHost));
   err(cudaMemcpy(weightsH, d_synH, sizeof(float)*128*128, cudaMemcpyDeviceToHost));

   //test
   printf("testing\n");
   float error = 0.0;

   err(cudaMemcpy(d_syn1, weights1, 64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn2, weights2, 4*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_synH, weightsH, 128*128*sizeof(float), cudaMemcpyHostToDevice));

   for (int i=0; i < num_test; ++i)
   {
      float* d_in;     
      float* d_layer1; 
      float* d_dlayer1;
      float* d_out;    
      float* out = (float*)malloc(Test[i].count*4*sizeof(float));
      err(cudaMalloc((void**)&d_in,     64*Test[i].count*sizeof(float)));
      err(cudaMalloc((void**)&d_layer1, Test[i].count*128*sizeof(float)));
      err(cudaMalloc((void**)&d_dlayer1,Test[i].count*128*sizeof(float)));
      err(cudaMalloc((void**)&d_out,    Test[i].count*4*sizeof(float)));

      err(cudaMemcpy(d_in, Test[i].Image, 64*Test[i].count*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_label, Test[i].Label, 4*sizeof(float), cudaMemcpyHostToDevice));

      //train<<<48,   125>>>(&d_in[6000*(iter%4)], &d_label[6000*(iter%4)], d_syn1, d_syn2, d_dsyn1, d_dsyn2, alpha);
      Fprop1<<<dim3(Test[i].count,16), dim3(128,4)>>>(d_in, d_syn1, d_layer1);
      for (int j=1; j < Test[i].count; ++j)
         FpropH<<<dim3(16,1), dim3(4,128)>>>(d_layer1, d_synH, j);
      Fprop2<<<dim3(Test[i].count,1), dim3(4,128)>>>(d_layer1, d_syn2, d_out);
      Dcalc2<<<dim3(1,Test[i].count), dim3(4,1)>>>(d_out, d_label);
      err(cudaMemcpy(out, d_out, sizeof(float)*4*Test[i].count, cudaMemcpyDeviceToHost));
      //Bprop2<<<dim3(Test[i].count,1), dim3(4,128)>>>(d_out, d_layer1, d_dsyn2, alpha);
      //Dcalc1<<<dim3(1,Test[i].count), dim3(128,1)>>>(d_out, d_dlayer1, d_syn2);
      //for (int i=Test[i].count-1; i > 0; --i)
      //   BpropH<<<dim3(32,1), dim3(4,128)>>>(d_layer1, d_dlayer1, d_dsynH, alpha, i);
      //Bprop1<<<dim3(Test[i].count,16), dim3(128,4)>>>(d_dlayer1, d_in, d_syn1, d_dsyn1, alpha);
      //err(cudaMemcpy(d_syn1, d_dsyn1, sizeof(float)*64*128, cudaMemcpyDeviceToDevice));
      //err(cudaMemcpy(d_syn2, d_dsyn2, sizeof(float)*128*4,    cudaMemcpyDeviceToDevice));
      //err(cudaMemcpy(d_synH, d_dsynH, sizeof(float)*128*128,   cudaMemcpyDeviceToDevice));

      int label_high = 0;
      float max = 0.0;
      for (int k=0; k < 4; ++k)
      {
         if (Test[i].Label[k] >= max)
         {
            max = Test[i].Label[k];
            label_high = k;
         }
      }

      int votes[4] = {0};
      for (int j=0; j < Test[i].count; ++j)
      {
         max = 0.0;
         int out_high = 0;
         for (int k=0; k < 4; ++k)
         {
            printf("%f, ", out[j*4 + k]);
            if (out[j*4 + k] > max)
            {
               max = out[j*4 + k];
               out_high = k;
            }
         }
         printf("\n");
         votes[out_high] += 1;
      }

      //printf("%d, %d, %d, %d\n", votes[0], votes[1], votes[2], votes[3]);
      int test_high = 0;
      int Max = 0;
      for (int k=0; k < 4; ++k)
      {
         if (votes[k] > Max)
         {
            Max = votes[k];
            test_high = k;
         }
      }

      //printf("%d, %d\n", test_high, label_high);
      if (test_high != label_high)
         error += 1.0/num_test;

      free(out);
      err(cudaFree(d_in));
      err(cudaFree(d_layer1));
      err(cudaFree(d_dlayer1));
      err(cudaFree(d_out));

   }
   //for (int i=0; i < 128; ++i)
   //{
   //   for (int j=0; j < 4; ++j)
   //   {
   //      printf("%f ", weights2[i*4 + j]);
   //   }
   //   printf("\n");
   //}
   //for (int i=0; i < Test.count; ++i)
   //{

   //   //reset layer states
   //   for (int j=0; j < 128; ++j)
   //      layer1[j] = 0.0;
   //   for (int j=0; j < 4; ++j)
   //      outs[j] = 0.0;

   //   // Forward pass
   //   //input to middle layer
   //   for (int j=0; j < Test.height; ++j)
   //   {
   //      for (int k=0; k < Test.width; ++k)
   //      {
   //         for (int l=0; l < 128; ++l)
   //         {
   //            layer1[l] += Test.Image[i*64 + j*28 + k] * weights1[j*28*128 + k*128 + l];
   //         }
   //      }
   //   }
   //   for (int j=0; j < 128; ++j)
   //      layer1[j] = sigmoid(layer1[j]);

   //   //middle to output layer
   //   for (int j=0; j < 128; ++j)
   //   {
   //      for (int k=0; k < 4; ++k)
   //      {
   //         outs[k] += layer1[j] * weights2[j*4 + k];
   //      }
   //   }
   //   for (int j=0; j < 4; ++j)
   //   {
   //      outs[j] = sigmoid(outs[j]);
   //      //printf("%f ", outs[j]);
   //   }
   //   //printf("\n");

   //   //sum up error
   //   for (int j=0; j < 4; ++j)
   //   {
   //      //printf("%f ", Test.Label[i*4 + j]);
   //      error += fabs(Test.Label[i*4 + j] - outs[j])/4.0;
   //   }
   //   //printf("\n");
   //}
   ////printf("Error: %f\n", error);
   //error /= Test.count;
   printf("Error: %f %%\n", error*100.0);

   //clean up data arrays
   //for (int i=0; i<Data.count; ++i)
   //{
   //   for (int j=0; j<28; ++j)
   //   {
   //      free(Data.Image[i][j]);
   //   }
   //   free(Data.Image[i]);
   //   free(Data.Label[i]);
   //}
   for (int d=0; d < num_data; ++d)
   {
      free(Data[d].Image);
      free(Data[d].Label);
   }
   for (int d=0; d < num_test; ++d)
   {
      free(Test[d].Image);
      free(Test[d].Label);
   }
   //for (int i=0; i<Test.count; ++i)
   //{
   //   for (int j=0; j<Test.height; ++j)
   //   {
   //      free(Test.Image[i][j]);
   //   }
   //   free(Test.Image[i]);
   //   free(Test.Label[i]);
   //}
   //for (int d=0; d < num_test; ++d)
   //{
   //   free(Test[d].Image);
   //   free(Test[d].Label);
   //}

   //err(cudaFree(d_in));
   err(cudaFree(d_label));
   //err(cudaFree(d_out));
   //err(cudaFree(d_layer1));
   //err(cudaFree(d_dlayer1));
   err(cudaFree(d_syn1));
   err(cudaFree(d_synH));
   err(cudaFree(d_syn2));
   err(cudaFree(d_dsyn1));
   err(cudaFree(d_dsynH));
   err(cudaFree(d_dsyn2));

   return EXIT_SUCCESS;
}
