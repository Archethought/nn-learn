#include "reader_ts.c"


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

//__device__ float tanh_(float x)
//{
//   // e**2x - 1
//   // ---------
//   // e**2x + 1
//   float exp2x =    exp(2.0*x);
//   return (exp2x - 1.0)/(exp2x + 1.0);
//}

__device__ float dtanh(float x)
{
   return 1.0 - x*x;
}

__global__ void Fprop1(const float* in, const float* syn1, float* layer1)
{
   int i = threadIdx.x;                         //128
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //64
   int k = blockIdx.x;                          //Data.count
   float x = 0.0;
   for (int j=0; j < 64; ++j)
      x += in[64*k + j] * syn1[j*128 + i];
   layer1[128*k + i] = x;
}
__global__ void LSTM1(float* layer1, float* lstm1, const float* gate1i, const float* gate1o, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //128
   float g_i = gate1i[128*offset + i];
   float g_f = 1.0 - g_i;
   float g_o = gate1o[128*offset + i];

   float i_t = tanh(layer1[128*offset + i]) * g_i;
   float i_p = 0.0;
   if (offset > 0)
      i_p = g_f * lstm1[128*(offset-1) + i];
   float sum = i_p + i_t;
   lstm1[128*offset + i] = sum;
   layer1[128*offset + i] = tanh(sum) * g_o;
}
__global__ void FpropH(float* layer1, const float* synH, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //128
   int j = blockDim.y*blockIdx.y + threadIdx.y; //128
      
   atomicAdd(&layer1[128*offset + j] , layer1[128*(offset-1) + i] * synH[i*128 + j]);
}
__global__ void Tanh(float* layer1)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 128
   layer1[i] = tanh(layer1[i]);
}
__global__ void Sigmoid(float* layer1)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 128
   layer1[i] = sigmoid(layer1[i]);
}
__global__ void Fprop2(const float* layer1, const float* syn2, float* out, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //4
   //int j = blockIdx.x;  //Data.count
   int k = blockDim.y*blockIdx.y + threadIdx.y; //128
   atomicAdd(&out[i], layer1[128*offset + k] * syn2[k*4 + i]);
}
__global__ void Ecalc2(float* out, const float* label)
{
   int i = threadIdx.x;                         //4
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count

   out[i] = label[i] - out[i];
}
__global__ void Dcalc2(float* out, const float* label)
{
   int i = threadIdx.x;                         //4
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count

   float x = label[i] - out[i];
   out[i] = x * dsigmoid(x);
}
__global__ void Bprop2(const float* out, const float* layer1, float* dsyn2, const int count, const float alpha)
{
   int i = blockDim.y*blockIdx.y + threadIdx.y; //128
   int j = blockDim.x*blockIdx.x + threadIdx.x; //4
   //int k = blockIdx.x;  //Data.count

   atomicAdd(&dsyn2[i*4 + j], out[j] * layer1[128*(count) + i] * alpha);
}
//__global__ void DcalcH(float* dlayer1, const float* layer1, const float* synH, const int offset)
//{
//   int i = blockDim.x*blockIdx.x + threadIdx.x; //128
//   int j = blockDim.y*blockIdx.y + threadIdx.y; //128
//
//   atomicAdd(&dlayer1[(offset-1)*128 + j], layer1[offset*128 + i] * synH[j*128 + i]);
//   float x = dlayer1[j*128 + i];
//   dlayer1[j*128 + i] = dtanh(x);
//}
__global__ void Dcalc1(const float* out, float* dlayer1, const float* syn2, const int count)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x;                         //128
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count

   float x = dlayer1[count*128 + i];
//#pragma unroll
   for (int k=0; k < 4; ++k)
      x += out[k] * syn2[i*4 + k];
   dlayer1[count*128 + i] = x * dtanh(x);
}
__global__ void BpropH(const float* layer1, float* dlayer1, const float* synH, float* dsynH, const float alpha, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //128
   int j = blockDim.y*blockIdx.y + threadIdx.y; //128

   atomicAdd(&dsynH[i*128 + j] , dlayer1[offset*128 + j] * layer1[(offset-1)*128 + i] * alpha);
   atomicAdd(&dlayer1[(offset-1)*128 + i] , layer1[offset*128 + j] * synH[i*128 + j]);
}
__global__ void BLSTM1(float* layer1, float* dlayer1, const float* lstm1, float* gate1i, float* gate1o)
{
   int i = threadIdx.x; //128
   int j = blockIdx.x;  //Data.count

   float e = dlayer1[j*128 + i];
   float C = lstm1[j*128 + i];
   float C_;
   if (j > 0)
      C_ = lstm1[(j-1)*128 + i];
   else
      C_ = 0.0;

   float o_o = gate1o[j*128 + i];
   float o_i = gate1o[j*128 + i];
   float o__i = C - (C_ * (1.0 - o_i));

   gate1o[j*128 + i] = o_o * e * dsigmoid(o_o); 
   gate1i[j*128 + i] = o_o * e * o_i * dsigmoid(o_i);
   layer1[j*128 + i] = o_o * e * o_i * o__i * dtanh(o__i);
}
__global__ void Bprop1(const float* dlayer1, const float* dlayer1i, const float* dlayer1o, const float* in, float* dsyn1, float* dsyn1i, float* dsyn1o, const float alpha)
{
   //int i = blockDim.y*blockIdx.y + threadIdx.y; //64
   int j = threadIdx.x;                         //128
   int k = blockIdx.x;                          //Data.count

   for (int i=0; i < 64; ++i)
   {
      atomicAdd(&dsyn1[i*128 + j]  , dlayer1[k*128 + j]  * in[k*64 + i] * alpha);
      atomicAdd(&dsyn1i[i*128 + j] , dlayer1i[k*128 + j] * in[k*64 + i] * alpha);
      atomicAdd(&dsyn1o[i*128 + j] , dlayer1o[k*128 + j] * in[k*64 + i] * alpha);
   }
}
//__global__ void Divide(float* syn, const float divisor)
//{
//   int i = blockDim.x*blockIdx.x + threadIdx.x;
//
//   syn[i] = syn[i]/divisor;
//}

//         Fprop1<<<dim3(Data[d].count,16), dim3(128,4)>>>(d_in, d_syn1, d_layer1);
//         for (int i=1; i < Data[d].count; ++i)
//            FpropH<<<dim3(32,1), dim3(4,128)>>>(d_layer1, d_synH, i);
//         Fprop2<<<dim3(Data[d].count,1), dim3(4,128)>>>(d_layer1, d_syn2, d_out);
//         Dcalc2<<<dim3(1,Data[d].count), dim3(4,1)>>>(d_out, d_label);
//         Bprop2<<<dim3(Data[d].count,1), dim3(4,128)>>>(d_out, d_layer1, d_dsyn2, alpha);
//         Dcalc1<<<dim3(1,Data[d].count), dim3(128,1)>>>(d_out, d_dlayer1, d_syn2);
//         for (int i=Data[d].count-1; i > 0; --i)
//            BpropH<<<dim3(32,1), dim3(4,128)>>>(d_layer1, d_dlayer1, d_dsynH, alpha, i);
//         Bprop1<<<dim3(Data[d].count,16), dim3(128,4)>>>(d_dlayer1, d_in, d_syn1, d_dsyn1, alpha);

//int wellDamn(float* weights1, float* weights1i, float* weights1o, float* weights2, float* weightsH, int iter)
//{
//   int damn = 0;
//   for (int x=0; x < 64*128; ++x)
//   {
//      if (isnan(weights1[x]))
//         damn |= 1;
//      if (isnan(weights1i[x]))
//         damn |= 2;
//      if (isnan(weights1o[x]))
//         damn |= 4;
//   }
//   for (int x=0; x < 128*4; ++x)
//   {
//      if (isnan(weights2[x]))
//         damn |= 16;
//   }
//   for (int x=0; x < 128*128; ++x)
//   {
//      if (isnan(weightsH[x]))
//         damn |= 8;
//   }
//   if(damn)
//      printf("%d, %d, damn\t", iter, damn);
//   return damn;
//}
//
//int godDamn(float* layer1, float* layer1i, float* layer1o, float* lstm1, float* dlayer1, float* out, int iter, int count)
//{
//   int damn = 0;
//#pragma omp parallel for
//   for (int x=0; x < count*128; ++x)
//   {
//      if (isnan(layer1[x]))
//         damn |= 1;
//      if (isnan(layer1i[x]))
//         damn |= 2;
//      if (isnan(layer1o[x]))
//         damn |= 4;
//      if (isnan(dlayer1[x]))
//         damn |= 8;
//      if (isnan(lstm1[x]))
//         damn |= 16;
//   }
//   for (int x=0; x < 4; ++x)
//   {
//      if (isnan(out[x]))
//         damn |= 32;
//   }
//   if(damn)
//      printf("%d, %d, goddamn\t", iter, damn);
//   return damn;
//}
//
//int godFuckingDamn(float* layer1, float* layer1i, float* layer1o, float* lstm1, float* dlayer1, float* out, int iter, int count)
//{
//   int damn = 0;
//#pragma omp parallel for
//   for (int x=0; x < count*128; ++x)
//   {
//      if (isnan(layer1[x]))
//      {
//         damn |= 1;
//         printf("L1:%d ", x%128);
//      }
//      if (isnan(layer1i[x]))
//      {
//         damn |= 2;
//         printf("L1i:%d ", x%128);
//      }
//      if (isnan(layer1o[x]))
//      {
//         damn |= 4;
//         printf("L1o:%d ", x%128);
//      }
//      if (isnan(dlayer1[x]))
//      {
//         damn |= 8;
//         printf("dL1:%d ", x%128);
//      }
//      if (isnan(lstm1[x]))
//      {
//         damn |= 16;
//         printf("LS1:%d ", x%128);
//      }
//   }
//   for (int x=0; x < 4; ++x)
//   {
//      if (isnan(out[x]))
//         damn |= 32;
//   }
//   if(damn)
//      printf("%d, %d, Fuck\n", iter, damn);
//   return damn;
//}

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
   int num_test = num_data/4/4*4;
   num_data -= num_test;

   if (num_data <= 0 || num_test <= 0)
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
   //for (int i=0; i < 5; ++i)
      //printf("%d\n", locations[i]);
   for (int j=0, k=0, l=0; j<4; ++j)
   {
      //int split = locations[j] + 0.7*(locations[j+1] - locations[j]);
      int split = (num_data)/4;
      //printf("%d\n", split);
//#pragma omp parallel for
      for (int i=locations[j]+1; i < locations[j+1]; ++i)
      {
         if (i-locations[j]-1 < split)
         {
            Data[k] = read_this(argv[i], genre[j]); //training data
            //printf("%d: %d [%f]\n", k, Data[k].count, Data[k].Image[Data[k].count-1]);
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

   float weights1i[64*128];    //input to middle layer weights
   float weights1o[64*128];    //input to middle layer weights
   float weights1[64*128];    //input to middle layer weights
   float weights2[128*4];     //middle to output layer weights
   float weightsH[128*128];   //propagation through time weights
   float alpha = atof(argv[argc-1]);

   //float* d_in;     err(cudaMalloc((void**)&d_in,     64*Data.count*sizeof(float)));
   //float* d_label;  err(cudaMalloc((void**)&d_label,  Data.count*4*sizeof(float)));
   //float* d_layer1; err(cudaMalloc((void**)&d_layer1, Data.count*128*sizeof(float)));
   //float* d_dlayer1;err(cudaMalloc((void**)&d_dlayer1,Data.count*128*sizeof(float)));
   //float* d_out;    err(cudaMalloc((void**)&d_out,    Data.count*4*sizeof(float)));
   float* d_syn1;   err(cudaMalloc((void**)&d_syn1,   64*128*sizeof(float)));
   float* d_dsyn1;  err(cudaMalloc((void**)&d_dsyn1,  64*128*sizeof(float)));
   float* d_syn1i;  err(cudaMalloc((void**)&d_syn1i,  64*128*sizeof(float)));
   float* d_dsyn1i; err(cudaMalloc((void**)&d_dsyn1i, 64*128*sizeof(float)));
   float* d_syn1o;  err(cudaMalloc((void**)&d_syn1o,  64*128*sizeof(float)));
   float* d_dsyn1o; err(cudaMalloc((void**)&d_dsyn1o, 64*128*sizeof(float)));
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
         weights1 [j*128 + k] = (float)rand()/(RAND_MAX/0.2) - 0.1;
         weights1i[j*128 + k] = (float)rand()/(RAND_MAX/0.2) - 0.1;
         weights1o[j*128 + k] = (float)rand()/(RAND_MAX/0.2) - 0.1;
      }
   }
   for (int i=0; i < 128; ++i)
   {
      for (int j=0; j < 4; ++j)
      {
         weights2[i*4 + j] = (float)rand()/(RAND_MAX/0.2) - 0.1;
      }
   }

   for (int i=0; i < 128; ++i)
   {
      for (int j=0; j < 128; ++j)
      {
         weightsH[i*128 + j] = (float)rand()/(RAND_MAX/0.2) - 0.1;
      }
   }


   //err(cudaMemcpy(d_in, Data.Image, 64*Data.count*sizeof(float), cudaMemcpyHostToDevice));
   //err(cudaMemcpy(d_label, Data.Label, 4*Data.count*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn1,  weights1, 64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1, weights1, 64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn1i, weights1i,64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1i,weights1i,64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn1o, weights1o,64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1o,weights1o,64*128*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn2,  weights2, 4*128*sizeof(float),  cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn2, weights2, 4*128*sizeof(float),  cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_synH,  weightsH, 128*128*sizeof(float),cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsynH, weightsH, 128*128*sizeof(float),cudaMemcpyHostToDevice));

   //cudaStream_t s[4];
   //cudaStreamCreate(&s[0]);
   //cudaStreamCreate(&s[1]);
   //cudaStreamCreate(&s[2]);
   //cudaStreamCreate(&s[3]);

   //train
   int iterations = atoi(argv[argc-2]);
   printf("training %d iterations\n", iterations);
   clock_t start_time = clock();
   for (int iter=0; iter<iterations; ++iter)
   {
//#pragma omp parallel for
      for (int d=0; d < num_data; ++d)
      {
         //printf("%d\t%f\n", Data[d].count, Data[d].Image[Data[d].count-1]);
         float* d_in;
         float* d_layer1;  float* layer1;
         float* d_layer1i; float* layer1i;
         float* d_layer1o; float* layer1o;
         float* d_dlayer1; float* dlayer1;
         float* d_lstm1;   float* lstm1;
         float* d_out;     float* out;

         err(cudaMalloc((void**)&d_in,     Data[d].count*64*sizeof(float)));
         err(cudaMalloc((void**)&d_layer1, Data[d].count*128*sizeof(float)));
         err(cudaMalloc((void**)&d_layer1i,Data[d].count*128*sizeof(float)));
         err(cudaMalloc((void**)&d_layer1o,Data[d].count*128*sizeof(float)));
         err(cudaMalloc((void**)&d_dlayer1,Data[d].count*128*sizeof(float)));
         err(cudaMalloc((void**)&d_lstm1,  Data[d].count*128*sizeof(float)));
         err(cudaMalloc((void**)&d_out,                  4*sizeof(float)));
         layer1  = (float*)malloc(Data[d].count*128*sizeof(float));
         layer1i = (float*)malloc(Data[d].count*128*sizeof(float));
         layer1o = (float*)malloc(Data[d].count*128*sizeof(float));
         dlayer1 = (float*)malloc(Data[d].count*128*sizeof(float));
         lstm1   = (float*)malloc(Data[d].count*128*sizeof(float));
         out     = (float*)malloc(              4*sizeof(float));

         //for (int i=0; i < Data[d].count*128; ++i)
         //   layer1[i] = 0;
         //for (int i=0; i < 4; ++i)
         //   out[i] = 0;

         //cudaDeviceSynchronize();
         err(cudaMemset(d_layer1,  0.0, Data[d].count*128*sizeof(float)));
         err(cudaMemset(d_layer1i, 0.0, Data[d].count*128*sizeof(float)));
         err(cudaMemset(d_layer1o, 0.0, Data[d].count*128*sizeof(float)));
         err(cudaMemset(d_dlayer1, 0.0, Data[d].count*128*sizeof(float)));
         err(cudaMemset(d_lstm1,   0.0, Data[d].count*128*sizeof(float)));
         err(cudaMemset(d_out,     0.0,               4  *sizeof(float)));

         err(cudaMemcpy(d_in,    Data[d].Image, 64*Data[d].count*sizeof(float), cudaMemcpyHostToDevice));
         err(cudaMemcpy(d_label, Data[d].Label, 4*sizeof(float), cudaMemcpyHostToDevice));

// forward pass
         Fprop1<<<dim3(Data[d].count,1), dim3(128,1)>>>(d_in, d_syn1, d_layer1);
         Tanh<<<Data[d].count, 128>>>(d_layer1);
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in Fprop1\n");
         Fprop1<<<dim3(Data[d].count,1), dim3(128,1)>>>(d_in, d_syn1i, d_layer1i);
         Sigmoid<<<Data[d].count, 128>>>(d_layer1i);
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in Fprop1i\n");
         Fprop1<<<dim3(Data[d].count,1), dim3(128,1)>>>(d_in, d_syn1o, d_layer1o);
         Sigmoid<<<Data[d].count, 128>>>(d_layer1o);
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in Fprop1o\n");
         LSTM1<<<1, 128>>>(d_layer1, d_lstm1, d_layer1i, d_layer1o, 0);
      //err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
      //cudaDeviceSynchronize();
      //if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in LSTM1(0)\n");
         for (int i=1; i < Data[d].count; ++i)
         {
            FpropH<<<dim3(1,1), dim3(1,128)>>>(d_layer1, d_synH, i);
      //if (iter >= 2)
      //{
      //err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
      //cudaDeviceSynchronize();
      //if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in FpropH(%d)\n", i);
      //}
            LSTM1<<<1, 128>>>(d_layer1, d_lstm1, d_layer1i, d_layer1o, i);
      //if (iter >= 2)
      //{
      //err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
      //cudaDeviceSynchronize();
      //if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in LSTM1(%d)\n", i);
      //}
         }
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in FpropH/LSTM1\n");
         Fprop2<<<dim3(4,1), dim3(1, 128)>>>(d_layer1, d_syn2, d_out, Data[d].count-1);
         Sigmoid<<<1, 4>>>(d_out);
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in Fprop2\n");

// backward pass
         Dcalc2<<<1, 4>>>(d_out, d_label);
         Bprop2<<<dim3(4,1), dim3(1,128)>>>(d_out, d_layer1, d_syn2, Data[d].count-1, alpha*10);
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1, d_dsyn1, sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1i,d_dsyn1i,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1o,d_dsyn1o,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights2, d_dsyn2, sizeof(float)*128*4,   cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weightsH, d_dsynH, sizeof(float)*128*128, cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in Dcalc2\n");
//      if (wellDamn(weights1, weights1i, weights1o, weights2, weightsH, iter)) printf("problem in Bprop2\n");
         Dcalc1<<<1, 128>>>(d_out, d_dlayer1, d_syn2, Data[d].count-1);
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in Dcalc1\n");
         for (int i=Data[d].count-1; i >= 1; i -= 1)
         {
            BpropH<<<dim3(1,128), dim3(128,1)>>>(d_layer1, d_dlayer1, d_synH, d_dsynH, alpha, i);
         }
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1, d_dsyn1, sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1i,d_dsyn1i,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1o,d_dsyn1o,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights2, d_dsyn2, sizeof(float)*128*4,   cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weightsH, d_dsynH, sizeof(float)*128*128, cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in BpropH\n");
//      if (wellDamn(weights1, weights1i, weights1o, weights2, weightsH, iter)) printf("problem in BpropH\n");
         BLSTM1<<<Data[d].count, 128>>>(d_layer1, d_dlayer1, d_lstm1, d_layer1i, d_layer1o);
         Bprop1<<<dim3(Data[d].count, 1), dim3(128, 1)>>>(d_layer1, d_layer1i, d_layer1o, d_in, d_dsyn1, d_dsyn1i, d_dsyn1o, alpha);
//      err(cudaMemcpy(layer1,  d_layer1,  Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1i, d_layer1i, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(layer1o, d_layer1o, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(dlayer1, d_dlayer1, Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(lstm1,   d_lstm1,   Data[d].count*128*sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(out,     d_out,                   4  *sizeof(float), cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1, d_dsyn1, sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1i,d_dsyn1i,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights1o,d_dsyn1o,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weights2, d_dsyn2, sizeof(float)*128*4,   cudaMemcpyDeviceToHost));
//      err(cudaMemcpy(weightsH, d_dsynH, sizeof(float)*128*128, cudaMemcpyDeviceToHost));
//      cudaDeviceSynchronize();
//      if (godDamn(layer1, layer1i, layer1o, lstm1, dlayer1, out, iter, Data[d].count)) printf("problem in BLSTM1\n");
//      if (wellDamn(weights1, weights1i, weights1o, weights2, weightsH, iter)) printf("problem in Bprop1\n");
         //Divide<<<128,128>>>(d_synH, (float)Data[d].count);
         //Divide<<<64,128>>>(d_syn1, (float)Data[d].count);

         err(cudaFree(d_in));
         err(cudaFree(d_layer1));
         err(cudaFree(d_layer1i));
         err(cudaFree(d_layer1o));
         err(cudaFree(d_dlayer1));
         err(cudaFree(d_lstm1));
         err(cudaFree(d_out));

         //cudaDeviceSynchronize();
         //free(out);
         free(layer1 );
         free(layer1i);
         free(layer1o);
         free(dlayer1);
         free(lstm1  );
         free(out    );
      }

      err(cudaMemcpy(d_syn1,  d_dsyn1, sizeof(float)*64*128,  cudaMemcpyDeviceToDevice));
      err(cudaMemcpy(d_syn1i, d_dsyn1i,sizeof(float)*64*128,  cudaMemcpyDeviceToDevice));
      err(cudaMemcpy(d_syn1o, d_dsyn1o,sizeof(float)*64*128,  cudaMemcpyDeviceToDevice));
      err(cudaMemcpy(d_syn2,  d_dsyn2, sizeof(float)*128*4,   cudaMemcpyDeviceToDevice));
      err(cudaMemcpy(d_synH,  d_dsynH, sizeof(float)*128*128, cudaMemcpyDeviceToDevice));
      //err(cudaMemcpy(weights1, d_dsyn1, sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights1i,d_dsyn1i,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights1o,d_dsyn1o,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights2, d_dsyn2, sizeof(float)*128*4,   cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weightsH, d_dsynH, sizeof(float)*128*128, cudaMemcpyDeviceToHost));

      //cudaDeviceSynchronize();

      //err(cudaMemcpy(d_syn1, weights1, sizeof(float)*64*128,  cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_syn1i,weights1i,sizeof(float)*64*128,  cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_syn1o,weights1o,sizeof(float)*64*128,  cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_syn2, weights2, sizeof(float)*128*4,   cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_synH, weightsH, sizeof(float)*128*128, cudaMemcpyHostToDevice));


   }
   //err(cudaMemcpy(weights1, d_syn1, sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
   //err(cudaMemcpy(weights2, d_syn2, sizeof(float)*128*4,   cudaMemcpyDeviceToHost));
   //err(cudaMemcpy(weightsH, d_synH, sizeof(float)*128*128, cudaMemcpyDeviceToHost));

   //cudaDeviceSynchronize();
   clock_t end_time = clock();
   double training_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
   printf("training time: %f\n", training_time);

   //test
   printf("testing\n");
   float error = 0.0;
   float numerical_error = 0.0;

   err(cudaMemcpy(d_syn1, weights1, 64*128*sizeof(float),  cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn2, weights2, 4*128*sizeof(float),   cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_synH, weightsH, 128*128*sizeof(float), cudaMemcpyHostToDevice));

   for (int d=0; d < num_test; ++d)
   {

      float* d_in;
      float* d_layer1;
      float* d_layer1i;
      float* d_layer1o;
      //float* d_dlayer1;
      float* d_lstm1;
      float* d_out;
      float layer1[Test[d].count*128];
      float* out = (float*)malloc(4*sizeof(float));
      err(cudaMalloc((void**)&d_in,     Test[d].count*64*sizeof(float)));
      err(cudaMalloc((void**)&d_layer1, Test[d].count*128*sizeof(float)));
      err(cudaMalloc((void**)&d_layer1i,Test[d].count*128*sizeof(float)));
      err(cudaMalloc((void**)&d_layer1o,Test[d].count*128*sizeof(float)));
      //err(cudaMalloc((void**)&d_dlayer1,Test[d].count*128*sizeof(float)));
      err(cudaMalloc((void**)&d_lstm1,  Test[d].count*128*sizeof(float)));
      err(cudaMalloc((void**)&d_out,                  4*sizeof(float)));

      for (int i=0; i < Test[d].count*128; ++i)
         layer1[i] = 0;
      for (int i=0; i < 4; ++i)
         out[i] = 0;

      err(cudaMemcpy(d_layer1, layer1, Test[d].count*128*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_layer1i,layer1, Test[d].count*128*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_layer1o,layer1, Test[d].count*128*sizeof(float), cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_dlayer1,layer1, Test[d].count*128*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_lstm1,  layer1, Test[d].count*128*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_out,    out,                  4  *sizeof(float), cudaMemcpyHostToDevice));

      err(cudaMemcpy(d_in,    Test[d].Image, 64*Test[d].count*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_label, Test[d].Label, 4*sizeof(float), cudaMemcpyHostToDevice));

      Fprop1<<<dim3(Test[d].count,1), dim3(128,1)>>>(d_in, d_syn1, d_layer1);
      Tanh<<<Test[d].count, 128>>>(d_layer1);
      Fprop1<<<dim3(Test[d].count,1), dim3(128,1)>>>(d_in, d_syn1i, d_layer1i);
      Sigmoid<<<Test[d].count, 128>>>(d_layer1i);
      Fprop1<<<dim3(Test[d].count,1), dim3(128,1)>>>(d_in, d_syn1o, d_layer1o);
      Sigmoid<<<Test[d].count, 128>>>(d_layer1o);
      LSTM1<<<1, 128>>>(d_layer1, d_lstm1, d_layer1i, d_layer1o, 0);
      for (int i=1; i < Test[d].count; ++i)
      {
         FpropH<<<dim3(1,1), dim3(1,128)>>>(d_layer1, d_synH, i);
         LSTM1<<<1, 128>>>(d_layer1, d_lstm1, d_layer1i, d_layer1o, i);
      }
      Fprop2<<<dim3(4,1), dim3(1, 128)>>>(d_layer1, d_syn2, d_out, Test[d].count-1);
      Sigmoid<<<1, 4>>>(d_out);

      //Fprop1<<<dim3(Test[d].count,64), dim3(128,1)>>>(d_in, d_syn1, d_layer1);
      //Tanh<<<Test[d].count, 128>>>(d_layer1);
      //Fprop1<<<dim3(Test[d].count,64), dim3(128,1)>>>(d_in, d_syn1i, d_layer1i);
      //Sigmoid<<<Test[d].count, 128>>>(d_layer1);
      //Fprop1<<<dim3(Test[d].count,64), dim3(128,1)>>>(d_in, d_syn1o, d_layer1o);
      //Sigmoid<<<Test[d].count, 128>>>(d_layer1);
      //for (int i=1; i < Test[d].count; ++i)
      //{
      //   LSTM1<<<1, 128>>>(d_layer1, d_lstm1, d_layer1i, d_layer1o, i);
      //   FpropH<<<dim3(32,1), dim3(4,128)>>>(d_layer1, d_synH, i);
      //}
      //Fprop2<<<1, 4>>>(d_layer1, d_synH, d_out, Test[d].count-1);
      //Sigmoid<<<1, 4>>>(d_out);


      err(cudaMemcpy(out, d_out, sizeof(float)*4,  cudaMemcpyDeviceToHost));

      //err(cudaMemcpy(weights1, d_dsyn1, sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights1i,d_dsyn1i,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights1o,d_dsyn1o,sizeof(float)*64*128,  cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weights2, d_dsyn2, sizeof(float)*128*4,   cudaMemcpyDeviceToHost));
      //err(cudaMemcpy(weightsH, d_dsynH, sizeof(float)*128*128, cudaMemcpyDeviceToHost));

      //err(cudaMemcpy(d_syn1, weights1, sizeof(float)*64*128,  cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_syn1i,weights1i,sizeof(float)*64*128,  cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_syn1o,weights1o,sizeof(float)*64*128,  cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_syn2, weights2, sizeof(float)*128*4,   cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_synH, weightsH, sizeof(float)*128*128, cudaMemcpyHostToDevice));

      err(cudaFree(d_in));
      err(cudaFree(d_layer1));
      err(cudaFree(d_layer1i));
      err(cudaFree(d_layer1o));
      //err(cudaFree(d_dlayer1));
      err(cudaFree(d_lstm1));
      err(cudaFree(d_out));

      cudaDeviceSynchronize();
      int label_high = -1;
      float max = -1.0;
      for (int k=0; k < 4; ++k)
      {
         if (Test[d].Label[k] >= max)
         {
            max = Test[d].Label[k];
            label_high = k;
         }
      }

      //int votes[4] = {0};
      max = -1.0;
      int out_high = -1;
      for (int k=0; k < 4; ++k)
      {
         if (d == 0 || d == num_test-1) printf("%f, ", out[k]);
         if (out[k] > max)
         {
            max = out[k];
            out_high = k;
         }
         numerical_error += fabs(out[k] - Test[d].Label[k])/num_test/4;
      }
      if (d == 0 || d == num_test-1) printf("\n");

      //printf("%d, %d\n", out_high, label_high);
      if (out_high != label_high)
         error += 1.0/num_test;

      free(out);
      //err(cudaFree(d_in));
      //err(cudaFree(d_layer1));
      //err(cudaFree(d_layer1i));
      //err(cudaFree(d_layer1o));
      //err(cudaFree(d_lstm1));
      //err(cudaFree(d_out));

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
   printf("Accuracy: %f %%\n", (1.0-error)*100.0);
   printf("Error:    %f %%\n", numerical_error*100.0);

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
