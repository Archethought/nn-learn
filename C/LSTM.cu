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
   return 4.0*x*(1.0-x);
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
__global__ void Sigmoid(float* layer1)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256
   layer1[i] = sigmoid(layer1[i]);
}
//__global__ void Tanh(float* layer1)
//{
//   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256
//   layer1[i] = tanh(layer1[i]);
//}

__global__ void Fprop1(const bool* drop, const float* in, const float* syn1, float* layer1)
{
   int i = threadIdx.x;                         //256
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //64
   int k = blockIdx.x;                          //Data.count

   if (drop[i])
   {
      float x = 0.0;
      for (int j=0; j < 64; ++j)
         x += in[64*k + j] * syn1[j*256 + i];
      layer1[256*k + i] = x;
   }
}
__global__ void LSTM1(const bool* drop, const float* layer1i, float* layer1o, float* lstm1, const float* gate1i, const float* gate1o, float* dgate1o, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //256

   if (drop[i])
   {
      float g_i = sigmoid(gate1i[256*offset + i]);
      float g_f = 1.0 - g_i;
      float g_o = sigmoid(gate1o[256*offset + i]);
      dgate1o[256*offset + i] = g_o;

      float i_c = g_i * tanh(layer1i[256*offset + i]);
      float i_p = 0.0;
      if (offset > 0)
         i_p = g_f * lstm1[256*(offset-1) + i];
      float sum = i_p + i_c;
      lstm1[256*offset + i] = sum;
      layer1o[256*offset + i] = tanh(sum) * g_o;
   }
}
__global__ void FpropH(const bool* drop, float* layer1, const float* layer1o, const float* synH, const int offset)
{
   int i = blockIdx.x; //256
   int j = threadIdx.x; //256

//   float x = 0.0;
//#pragma unroll
//   for (int i=0; i < 256; ++i)
//      x +=  layer1o[256*(offset-1) + i] * synH [i*256 + j];
//   layer1[256*offset + j] = x;
   if (drop[i] && drop[j])
      atomicAdd(&layer1[256*offset + j] , layer1o[256*(offset-1) + i] * synH [i*256 + j]);
}
__global__ void Fprop2(const bool* drop, const float* layer1, const float* syn2, float* out)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //4
   int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count
   //int k = blockDim.y*blockIdx.y + threadIdx.y; //256
   //atomicAdd(&out[j*4 + i], layer1[256*j + k] * syn2[k*4 + i]);

   float x = 0.0;
#pragma unroll
   for (int k=0; k < 256; ++k)
   {
      if (drop[i])
         x += layer1[256*j + k] * syn2[k*4 + i];
   }
   out[j*4 + i] = sigmoid(x);
}
__global__ void Ecalc2(float* out, const float* label)
{
   int i = threadIdx.x;    //4
   int j = blockIdx.x;     //Data.count

   out[i] = label[i] - out[4*j + i];
}
__global__ void Dcalc2(float* out, const float* label)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //4
   int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count

   float x = label[i] - out[4*j + i];
   out[4*j + i] = x * dsigmoid(out[4*j + i]);
}
__global__ void Bprop2(const bool* drop, const float* out, const float* layer1, float* dsyn2, const float alpha)
{
   int i = blockDim.y*blockIdx.y + threadIdx.y; //256
   int j = threadIdx.x; //4
   int k = blockIdx.x;  //Data.count

   if (drop[i])
      atomicAdd(&dsyn2[i*4 + j], out[4*k + j] * layer1[256*k + i] * alpha);
}
__global__ void Dcalc1(const bool* drop, const float* out, float* dlayer1, const float* syn2)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x; //Data.count
   int l = gridDim.x - 1; //last output

   if (drop[i])
   {
      float x = 0.0;
      float y = 0.0;
//#pragma unroll
      for (int k=0; k < 4; ++k)
      {
         x += out[j*4 + k] * syn2[i*4 + k];
         y += out[l*4 + k] * syn2[i*4 + k];
      }
      dlayer1[j*256 + i] = (x+y)/2;
   }
}
__global__ void BpropH(const bool* drop, const float* dlstm1, const float* gate1i, const float* gate1o, const float* dgate1o, const float* lstm1, float* dsynH, float* dsynHi, float* dsynHo, const float alpha)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //256
   int offset = blockIdx.y+1; //Data.count-1

   if (drop[i])
   {
      float d_o = tanh(lstm1[(offset-1)*256 + i]) * dgate1o[(offset-1)*256 + i];
      atomicAdd(&dsynH [i*256 + j] , dlstm1[offset*256 + j] * d_o * alpha);
      atomicAdd(&dsynHi[i*256 + j] , gate1i[offset*256 + j] * d_o * alpha);
      atomicAdd(&dsynHo[i*256 + j] , gate1o[offset*256 + j] * d_o * alpha);
   }
}
__global__ void DcalcH(const bool* drop, const float* layer1i, float* dlayer1, const float* synH, const int offset)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //256

   if (drop[i] && drop[j])
   {
      //float x = 0.0;
      //for (int j=0; j < 256; ++j)
      //   x += layer1i[offset*256 + j] * synH[i*256 + j];
      //dlayer1[(offset-1)*256 + i] += x;
      atomicAdd(&dlayer1[(offset-1)*256 + i] , layer1i[offset*256 + j] * synH[i*256 + j]);
   }
}
__global__ void BLSTM1(const bool* drop, const float* layer1i, const float* layer1o, const float* dlayer1, const float* lstm1, float* gate1i, float* gate1o)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //Data.count

   if (drop[i])
   {
      float d_o = tanh(lstm1[256*j + i])   * dlayer1[j*256 + i];
      float d_i = tanh(layer1i[256*j + i]) * layer1o[j*256 + i];
      float d_f = 0.0;
      if (j == 0)
         d_f = lstm1[256*(j-1) + i]     * layer1o[j*256 + i];;
      gate1o[256*j + i] = d_o * dsigmoid(sigmoid(gate1o[j*256 + i]));
      gate1i[256*j + i] = (d_i - d_f) * dsigmoid(sigmoid(gate1i[256*j + i]));
   }
}
__global__ void BLSTMH(const bool* drop, const float* layer1i, float* layer1o, const float* dlayer1, float* dlstm1, const float* lstm1, const float* gate1i, const float* gate1o, const int offset, const bool last)
{
   int i = threadIdx.x; //256
   int j = offset;

   if (drop[i])
   {
      float e_c = dlayer1[j*256 + i];

      float e_s = sigmoid(gate1o[256*j + i]) * dtanh(layer1o[256*j + i]) * e_c;
      if (!last)
         e_s += (1.0 - sigmoid(gate1i[256*(j+1) + i])) * layer1o[256*(j+1) + i];
      layer1o[256*j + i] = e_s;
      float d_c = sigmoid(gate1i[256*j + i]) * dtanh(layer1i[256*j + i]) * e_s;
      dlstm1 [256*j + i] = d_c;
   }
}
__global__ void Bprop1(const bool* drop, const float* dlstm1, const float* gate1i, const float* gate1o, const float* in, float* dsyn1, float* dsyn1i, float* dsyn1o, const float alpha)
{
   //int i = blockDim.y*blockIdx.y + threadIdx.y; //64
   int j = threadIdx.x;                         //256
   int k = blockIdx.x;                          //Data.count

   if (drop[j])
   {
      for (int i=0; i < 64; ++i)
      {
         if (dlstm1[k*256 + j] != 0.0)
            atomicAdd(&dsyn1[i*256 + j]  , dlstm1[k*256 + j] * in[k*64 + i] * alpha);
         if (gate1i[k*256 + j] != 0.0)
            atomicAdd(&dsyn1i[i*256 + j] , gate1i[k*256 + j] * in[k*64 + i] * alpha);
         if (gate1o[k*256 + j] != 0.0)
            atomicAdd(&dsyn1o[i*256 + j] , gate1o[k*256 + j] * in[k*64 + i] * alpha);
      }
   }
}


void pickle(float* syn1, float* syn1i, float* syn1o, float* synH, float* synHi, float* synHo, float* syn2, char* filename)
{
   int er = 0;
   FILE* outfile = fopen(filename, "wb");
   er += fwrite(syn1, sizeof(float), 64*256, outfile);
   er += fwrite(syn1i,sizeof(float), 64*256, outfile);
   er += fwrite(syn1i,sizeof(float), 64*256, outfile);
   er += fwrite(synH, sizeof(float), 256*256, outfile);
   er += fwrite(synHi,sizeof(float), 256*256, outfile);
   er += fwrite(synHo,sizeof(float), 256*256, outfile);
   er += fwrite(syn2, sizeof(float), 256*4, outfile);
   printf("%d\n", er);
   fclose(outfile);
}
void unpickle(float* syn1, float* syn1i, float* syn1o, float* synH, float* synHi, float* synHo, float* syn2, char* filename)
{
   int er = 0;
   FILE* infile = fopen(filename, "rb");
   er += fread(syn1, sizeof(float), 64*256, infile);
   er += fread(syn1i,sizeof(float), 64*256, infile);
   er += fread(syn1o,sizeof(float), 64*256, infile);
   er += fread(synH, sizeof(float), 256*256, infile);
   er += fread(synHi,sizeof(float), 256*256, infile);
   er += fread(synHo,sizeof(float), 256*256, infile);
   er += fread(syn2, sizeof(float), 256*4, infile);
   printf("%d\n", er);
   fclose(infile);
}


int main(int argc, char** argv)
{
   int num_data = 0;;
   int num_test = 0;

   struct data* Data = NULL;
   struct data* Test = NULL;

   //float* d_in;     err(cudaMalloc((void**)&d_in,     64*Data.count*sizeof(float)));
   //float* d_label;  err(cudaMalloc((void**)&d_label,  Data.count*4*sizeof(float)));
   //float* d_layer1; err(cudaMalloc((void**)&d_layer1, Data.count*256*sizeof(float)));
   //float* d_dlayer1;err(cudaMalloc((void**)&d_dlayer1,Data.count*256*sizeof(float)));
   //float* d_out;    err(cudaMalloc((void**)&d_out,    Data.count*4*sizeof(float)));
   float* d_syn1=NULL;   err(cudaMalloc((void**)&d_syn1,   64*256*sizeof(float)));
   float* d_dsyn1=NULL;  err(cudaMalloc((void**)&d_dsyn1,  64*256*sizeof(float)));
   float* d_syn1i=NULL;  err(cudaMalloc((void**)&d_syn1i,  64*256*sizeof(float)));
   float* d_dsyn1i=NULL; err(cudaMalloc((void**)&d_dsyn1i, 64*256*sizeof(float)));
   float* d_syn1o=NULL;  err(cudaMalloc((void**)&d_syn1o,  64*256*sizeof(float)));
   float* d_dsyn1o=NULL; err(cudaMalloc((void**)&d_dsyn1o, 64*256*sizeof(float)));
   float* d_synH=NULL;   err(cudaMalloc((void**)&d_synH,   256*256*sizeof(float)));
   float* d_dsynH=NULL;  err(cudaMalloc((void**)&d_dsynH,  256*256*sizeof(float)));
   float* d_synHi=NULL;  err(cudaMalloc((void**)&d_synHi,  256*256*sizeof(float)));
   float* d_dsynHi=NULL; err(cudaMalloc((void**)&d_dsynHi, 256*256*sizeof(float)));
   float* d_synHo=NULL;  err(cudaMalloc((void**)&d_synHo,  256*256*sizeof(float)));
   float* d_dsynHo=NULL; err(cudaMalloc((void**)&d_dsynHo, 256*256*sizeof(float)));
   float* d_syn2=NULL;   err(cudaMalloc((void**)&d_syn2,   256*4*sizeof(float)));
   float* d_dsyn2=NULL;  err(cudaMalloc((void**)&d_dsyn2,  256*4*sizeof(float)));
   float* d_label=NULL;  err(cudaMalloc((void**)&d_label,  4*sizeof(float)));

   float weights1[64*256];    //input to middle layer weights
   float weights1i[64*256];    //input to middle layer weights
   float weights1o[64*256];    //input to middle layer weights
   float weightsH[256*256];   //propagation through time weights
   float weightsHi[256*256];   //propagation through time weights
   float weightsHo[256*256];   //propagation through time weights
   float weights2[256*4];     //middle to output layer weights

   cudaStream_t s[3];
   cudaStreamCreate(&s[0]);
   cudaStreamCreate(&s[1]);
   cudaStreamCreate(&s[2]);
   //cudaStreamCreate(&s[3]);

   if (!strcmp(argv[argc-1], "r"))
   {
      num_test = (argc-2)/2;
      Test = (struct data*)malloc(num_test*sizeof(struct data));
      for (int i=1, k=0; i < argc-2; i += 2, ++k)
      {
         Test[k] = read_known(argv[i+1], argv[i]);
      }
      unpickle(weights1, weights1i, weights1o, weightsH, weightsHi, weightsHo, weights2, argv[argc-2]);
   }

   else
   {
      int num_args = !strcmp(argv[argc-1], "w") ? argc-2 : argc;
      const char* genre[4] = {"classical", "jazz", "metal", "pop"};
      int locations[5] = {0};
      num_data = num_args-7;
      num_test = num_data/4/4*4;
      num_data -= num_test;

      if (num_data <= 0 || num_test <= 0)
      {  printf("too few data\n"); exit(1); }

      printf("reading data\n");
      Data = (struct data*)malloc(num_data*sizeof(struct data));
      Test = (struct data*)malloc(num_test*sizeof(struct data));
      for (int i=1; i < num_args-2; ++i)
      {
         for (int j=0; j < 4; ++j)
         {
            if (!strcmp(argv[i], genre[j]))
               locations[j] = i;
         }
      }
      locations[4] = num_args-2;
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
               Data[k] = read_known(argv[i], genre[j]); //training data
               //printf("%d: %d [%f]\n", k, Data[k].count, Data[k].Image[Data[k].count-1]);
               ++k;
            }
            else
            {
               Test[l] = read_known(argv[i], genre[j]); //testing data
               ++l;
            }
            //printf("%d, %d, %d:%x, %d:%x\n", i, i<split, k, Data[k-1].Image, l, Test[l-1].Image);
         }
      }

      //Initialize weights to random values
      //printf("randomizing initial weights\n");
      srand(112992); //make the random values the same each time
      for (int j=0; j < 64; ++j)
      {
         for (int k=0; k < 256; ++k)
         {
            weights1 [j*256 + k] = (float)rand()/(RAND_MAX/0.2) - 0.1;
            weights1i[j*256 + k] = (float)rand()/(RAND_MAX/0.2) - 0.1;
            weights1o[j*256 + k] = (float)rand()/(RAND_MAX/0.2) - 0.1;
         }
      }
      for (int i=0; i < 256; ++i)
      {
         for (int j=0; j < 4; ++j)
         {
            weights2[i*4 + j] = (float)rand()/(RAND_MAX/0.2) - 0.1;
         }
      }

      for (int i=0; i < 256; ++i)
      {
         for (int j=0; j < 256; ++j)
         {
            weightsH[i*256 + j]  = (float)rand()/(RAND_MAX/0.2) - 0.1;
            weightsHi[i*256 + j] = (float)rand()/(RAND_MAX/0.2) - 0.1;
            weightsHo[i*256 + j] = (float)rand()/(RAND_MAX/0.2) - 0.1;
         }
      }


      //err(cudaMemcpy(d_in, Data.Image, 64*Data.count*sizeof(float), cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_label, Data.Label, 4*Data.count*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn1,  weights1, 64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn1, weights1, 64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn1i, weights1i,64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn1i,weights1i,64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn1o, weights1o,64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn1o,weights1o,64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_synH,  weightsH, 256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsynH, weightsH, 256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_synHi, weightsHi,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsynHi,weightsHi,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_synHo, weightsHo,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsynHo,weightsHo,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn2,  weights2, 4*256*sizeof(float),  cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn2, weights2, 4*256*sizeof(float),  cudaMemcpyHostToDevice));

      //train
      float alpha = atof(argv[num_args-1]);
      bool* dropout = (bool*)malloc(256*sizeof(bool));
      bool* d_drop;  err(cudaMalloc((void**)&d_drop, 256*sizeof(bool)));
      int iterations = atoi(argv[num_args-2]);
      printf("training %d iterations\n", iterations);
      clock_t start_time = clock();
      for (int iter=0; iter<iterations; ++iter)
      {
         //printf("iteration %d\n", iter);
         for (int d=0; d < num_data; ++d)
         {
            //randomize dropped nodes
            memset(dropout, false, 256*sizeof(bool));
            for (int i=0; i < 256/2; ++i)
            {
               int x;
               do{
                  x = rand()%(256);
               } while (dropout[x]);
               dropout[x] = true;
               //printf("%d,", x);
            }
            err(cudaMemcpy(d_drop, dropout, 256*sizeof(bool), cudaMemcpyHostToDevice));
            //printf("\n");

            //printf("-\n");
            //printf("%d\t%f\n", Data[d].count, Data[d].Image[Data[d].count-1]);
            float* d_in;
            //float* d_layer1;
            //float* layer1;
            float* d_layer1i; //float* layer1i;
            float* d_layer1o; //float* layer1o;
            float* d_gate1i;  //float* gate1i;
            float* d_gate1o;  //float* gate1o;
            float* d_dgate1o; //float* gate1o;
            float* d_dlayer1; //float* dlayer1;
            float* d_dlstm1;  //float* dlstm1;
            float* d_lstm1;   //float* lstm1;
            float* d_out;     //float* out;

            err(cudaMalloc((void**)&d_in,     Data[d].count*64*sizeof(float)));
            //err(cudaMalloc((void**)&d_layer1,Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_layer1i,Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_layer1o,Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_gate1i, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_gate1o, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_dgate1o, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_dlayer1,Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_dlstm1, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_lstm1,  Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_out,    Data[d].count*4*sizeof(float)));
            //layer1  = (float*)malloc(Data[d].count*256*sizeof(float));
            //layer1i = (float*)malloc(Data[d].count*256*sizeof(float));
            //layer1o = (float*)malloc(Data[d].count*256*sizeof(float));
            //gate1i  = (float*)malloc(Data[d].count*256*sizeof(float));
            //gate1o  = (float*)malloc(Data[d].count*256*sizeof(float));
            //dlayer1 = (float*)malloc(Data[d].count*256*sizeof(float));
            //lstm1   = (float*)malloc(Data[d].count*256*sizeof(float));
            //out     = (float*)malloc(Data[d].count*4*sizeof(float));

            //for (int i=0; i < Data[d].count*256; ++i)
            //   layer1[i] = 0;
            //for (int i=0; i < 4; ++i)
            //   out[i] = 0;

            //cudaDeviceSynchronize();
            //err(cudaMemset(d_layer1,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_layer1i, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_layer1o, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_gate1i,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_gate1o,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_dgate1o, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_dlayer1, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_dlstm1,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_lstm1,   0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_out,     0.0, Data[d].count*4  *sizeof(float)));

            err(cudaMemcpy(d_in,    Data[d].Image, 64*Data[d].count*sizeof(float), cudaMemcpyHostToDevice));
            err(cudaMemcpy(d_label, Data[d].Label, 4*sizeof(float), cudaMemcpyHostToDevice));

      // forward pass
            Fprop1<<<Data[d].count, 256 ,0, s[0]>>>(d_drop, d_in, d_syn1, d_layer1i);
            Fprop1<<<Data[d].count, 256 ,0, s[1]>>>(d_drop, d_in, d_syn1i, d_gate1i);
            Fprop1<<<Data[d].count, 256 ,0, s[2]>>>(d_drop, d_in, d_syn1o, d_gate1o);
            LSTM1<<<1, 256>>>(d_drop, d_layer1i, d_layer1o, d_lstm1, d_gate1i, d_gate1o, d_dgate1o, 0);
            for (int i=1; i < Data[d].count; ++i)
            {
               FpropH<<<256, 256 ,0, s[0]>>>(d_drop, d_layer1i, d_layer1o, d_synH,  i);
               FpropH<<<256, 256 ,0, s[1]>>>(d_drop, d_gate1i,  d_layer1o, d_synHi, i);
               FpropH<<<256, 256 ,0, s[2]>>>(d_drop, d_gate1o,  d_layer1o, d_synHo, i);
               LSTM1<<<1, 256>>>(d_drop, d_layer1i, d_layer1o, d_lstm1, d_gate1i, d_gate1o, d_dgate1o, i);
            }
            Fprop2<<<dim3(1, Data[d].count), dim3(4, 1)>>>(d_drop, d_layer1o, d_syn2, d_out);

      // backward pass
            Dcalc2<<<dim3(1, Data[d].count), dim3(4, 1)>>>(d_out, d_label);
            Bprop2<<<dim3(Data[d].count,1), dim3(4,256)>>>(d_drop, d_out, d_layer1o, d_dsyn2, alpha/Data[d].count);
            Dcalc1<<<Data[d].count, 256>>>(d_drop, d_out, d_dlayer1, d_syn2);
            bool last = true;
            for (int i=Data[d].count-1; i >= 1; i -= 1)
            {
               BLSTMH<<<1, 256>>>(d_drop, d_layer1i, d_layer1o, d_dlayer1, d_dlstm1, d_lstm1, d_gate1i, d_gate1o, i, last);
               DcalcH<<<256, 256>>>(d_drop, d_layer1i, d_dlayer1, d_synH, i);

               //err(cudaMemcpy(layer1, d_dlstm1, sizeof(float)*Data[d].count*256, cudaMemcpyDeviceToHost));
               //bool thing = false;
               //for (int j=0; j < Data[d].count*256; ++j)
               //{
               //   if (isnan(layer1[j]))
               //      thing = true;
               //}
               //if (thing)
               //{
               //   printf("problem at i = %d/%d\n", i, Data[d].count-1);
               //   break;
               //}
               last = false;
            }
            BLSTMH<<<1, 256>>>(d_drop, d_layer1i, d_layer1o, d_dlayer1, d_dlstm1, d_lstm1, d_gate1i, d_gate1o, 0, false);

            BLSTM1<<<Data[d].count, 256>>>(d_drop, d_layer1i, d_layer1o, d_dlayer1, d_lstm1, d_layer1i, d_layer1o);
            BpropH<<<dim3(256, Data[d].count-2), dim3(256, 1) ,0, s[0]>>>(d_drop, d_dlstm1, d_gate1i, d_gate1o, d_dgate1o, d_lstm1, d_dsynH, d_dsynHi, d_dsynHo, alpha/Data[d].count);
            Bprop1<<<Data[d].count, 256 ,0, s[1]>>>(d_drop, d_dlstm1, d_gate1i, d_gate1o, d_in, d_dsyn1, d_dsyn1i, d_dsyn1o, alpha/Data[d].count);

            err(cudaFree(d_in));
            err(cudaFree(d_layer1i));
            err(cudaFree(d_layer1o));
            err(cudaFree(d_gate1i));
            err(cudaFree(d_gate1o));
            err(cudaFree(d_dgate1o));
            err(cudaFree(d_dlayer1));
            err(cudaFree(d_lstm1));
            err(cudaFree(d_dlstm1));
            err(cudaFree(d_out));

            cudaDeviceSynchronize();
         }

         //printf("\n");

         err(cudaMemcpy(d_syn1,  d_dsyn1, sizeof(float)*64*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_syn1i, d_dsyn1i,sizeof(float)*64*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_syn1o, d_dsyn1o,sizeof(float)*64*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_synH,  d_dsynH, sizeof(float)*256*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_synHi, d_dsynHi,sizeof(float)*256*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_synHo, d_dsynHo,sizeof(float)*256*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_syn2,  d_dsyn2, sizeof(float)*256*4,   cudaMemcpyDeviceToDevice));

         err(cudaMemcpy(weights1,  d_dsyn1, sizeof(float)*64*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weights1i, d_dsyn1i,sizeof(float)*64*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weights1o, d_dsyn1o,sizeof(float)*64*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weightsH,  d_dsynH, sizeof(float)*256*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weightsHi, d_dsynHi,sizeof(float)*256*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weightsHo, d_dsynHo,sizeof(float)*256*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weights2,  d_dsyn2, sizeof(float)*256*4,   cudaMemcpyDeviceToHost));

         //int er = 0;
         //for (int i=0; i < 64*256; ++i)
         //{
         //   if (isnan(weights1[i]))
         //      er |= 1;
         //   if (isnan(weights1i[i]))
         //      er |= 2;
         //   if (isnan(weights1o[i]))
         //      er |= 4;
         //}
         //for (int i=0; i < 256*256; ++i)
         //{
         //   if (isnan(weightsH[i]))
         //      er |= 8;
         //   if (isnan(weightsHi[i]))
         //      er |= 16;
         //   if (isnan(weightsHo[i]))
         //      er |= 32;
         //}
         //for (int i=0; i < 256*4; ++i)
         //{
         //   if (isnan(weights2[i]))
         //      er |= 64;
         //}

         //if (er)
         //   printf("%x\n", er);

      }

      //cudaDeviceSynchronize();
      clock_t end_time = clock();
      double training_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
      printf("training time: %f\n", training_time);

      free(dropout);
      err(cudaFree(d_drop));
   }

   //test
   printf("testing\n");
   float error = 0.0;
   float numerical_error = 0.0;

   bool* dropout = (bool*)malloc(256*sizeof(bool));
   bool* d_drop;  err(cudaMalloc((void**)&d_drop, 256*sizeof(bool)));

   for (int d=0; d < num_test; ++d)
   {
      float* d_in;
      //float* d_layer1;  //float* layer1;
      float* d_layer1i; //float* layer1i;
      float* d_layer1o; //float* layer1o;
      float* d_gate1i;  //float* gate1i;
      float* d_gate1o;  //float* gate1o;
      float* d_dgate1o; //float* gate1o;
      //float* d_dlayer1; //float* dlayer1;
      //float* d_dlstm1;  //float* dlstm1;
      float* d_lstm1;   //float* lstm1;
      float* d_out;     //float* out;

      err(cudaMalloc((void**)&d_in,     Test[d].count*64*sizeof(float)));
      //err(cudaMalloc((void**)&d_layer1,Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_layer1i,Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_layer1o,Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_gate1i, Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_gate1o, Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_dgate1o,Test[d].count*256*sizeof(float)));
      //err(cudaMalloc((void**)&d_dlayer1,Test[d].count*256*sizeof(float)));
      //err(cudaMalloc((void**)&d_dlstm1, Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_lstm1,  Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_out,    Test[d].count*4*sizeof(float)));

      err(cudaMemset(d_layer1i, 0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_layer1o, 0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_gate1i,  0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_gate1o,  0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_dgate1o, 0.0, Test[d].count*256*sizeof(float)));
      //err(cudaMemset(d_dlayer1, 0.0, Test[d].count*256*sizeof(float)));
      //err(cudaMemset(d_dlstm1,  0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_lstm1,   0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_out,     0.0, Test[d].count*4  *sizeof(float)));

      err(cudaMemcpy(d_in,    Test[d].Image, 64*Test[d].count*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_label, Test[d].Label, 4*sizeof(float), cudaMemcpyHostToDevice));

// forward pass
      //randomize dropped nodes
      memset(dropout, false, 256*sizeof(bool));
      for (int i=0; i < 256/2; ++i)
      {
         int x;
         do
         {
            x = rand()%(256);
         } while (dropout[x]);
         dropout[x] = true;
         //printf("%d,", x);
      }
      err(cudaMemcpy(d_drop, dropout, 256*sizeof(bool), cudaMemcpyHostToDevice));

      Fprop1<<<Test[d].count, 256 ,0, s[0]>>>(d_drop, d_in, d_syn1, d_layer1i);
      Fprop1<<<Test[d].count, 256 ,0, s[1]>>>(d_drop, d_in, d_syn1i, d_gate1i);
      Fprop1<<<Test[d].count, 256 ,0, s[2]>>>(d_drop, d_in, d_syn1o, d_gate1o);
      LSTM1<<<1, 256>>>(d_drop, d_layer1i, d_layer1o, d_lstm1, d_gate1i, d_gate1o, d_dgate1o, 0);
      for (int i=1; i < Test[d].count; ++i)
      {
         FpropH<<<256, 256 ,0, s[0]>>>(d_drop, d_layer1i, d_layer1o, d_synH,  i);
         FpropH<<<256, 256 ,0, s[1]>>>(d_drop, d_gate1i,  d_layer1o, d_synHi, i);
         FpropH<<<256, 256 ,0, s[2]>>>(d_drop, d_gate1o,  d_layer1o, d_synHo, i);
         LSTM1<<<1, 256>>>(d_drop, d_layer1i, d_layer1o, d_lstm1, d_gate1i, d_gate1o, d_dgate1o, i);
      }
      Fprop2<<<dim3(1, Test[d].count), dim3(4, 1)>>>(d_drop, d_layer1o, d_syn2, d_out);

      float* out = (float*)malloc(4*sizeof(float));

      err(cudaMemcpy(out, &d_out[(Test[d].count-1)*4], sizeof(float)*4,  cudaMemcpyDeviceToHost));


      err(cudaFree(d_in));
      err(cudaFree(d_layer1i));
      err(cudaFree(d_layer1o));
      err(cudaFree(d_gate1i));
      err(cudaFree(d_gate1o));
      err(cudaFree(d_dgate1o));
      err(cudaFree(d_lstm1));
      err(cudaFree(d_out));

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
   }

   free(dropout);
   err(cudaFree(d_drop));
   //for (int i=0; i < 256; ++i)
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
   //   for (int j=0; j < 256; ++j)
   //      layer1[j] = 0.0;
   //   for (int j=0; j < 4; ++j)
   //      outs[j] = 0.0;

   //   // Forward pass
   //   //input to middle layer
   //   for (int j=0; j < Test.height; ++j)
   //   {
   //      for (int k=0; k < Test.width; ++k)
   //      {
   //         for (int l=0; l < 256; ++l)
   //         {
   //            layer1[l] += Test.Image[i*64 + j*28 + k] * weights1[j*28*256 + k*256 + l];
   //         }
   //      }
   //   }
   //   for (int j=0; j < 256; ++j)
   //      layer1[j] = sigmoid(layer1[j]);

   //   //middle to output layer
   //   for (int j=0; j < 256; ++j)
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

   //err(cudaFree(d_in));
   err(cudaFree(d_label));
   //err(cudaFree(d_out));
   //err(cudaFree(d_layer1));
   //err(cudaFree(d_dlayer1));
   err(cudaFree(d_syn1));
   err(cudaFree(d_syn1i));
   err(cudaFree(d_syn1o));
   err(cudaFree(d_synH));
   err(cudaFree(d_synHi));
   err(cudaFree(d_synHo));
   err(cudaFree(d_syn2));
   err(cudaFree(d_dsyn1));
   err(cudaFree(d_dsyn1i));
   err(cudaFree(d_dsyn1o));
   err(cudaFree(d_dsynH));
   err(cudaFree(d_dsynHi));
   err(cudaFree(d_dsynHo));
   err(cudaFree(d_dsyn2));

   if (!strcmp(argv[argc-1], "w"))
   {
      printf("writing to %s\n", argv[argc-2]);
      pickle(weights1, weights1i, weights1o, weightsH, weightsHi, weightsHo, weights2, argv[argc-2]);
   }

   return EXIT_SUCCESS;
}
