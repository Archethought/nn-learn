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
__global__ void Sigmoid(float* layer2)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256
   layer2[i] = sigmoid(layer2[i]);
}
__global__ void Tanh(float* layer2)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256
   layer2[i] = tanh(layer2[i]);
}

__global__ void Fprop1(const float* in, const float* syn2, float* layer2)
{
   int i = threadIdx.x;                         //256
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //64
   int k = blockIdx.x;                          //Data.count

   float x = 0.0;
   for (int j=0; j < 64; ++j)
      x += in[64*k + j] * syn2[j*256 + i];
   layer2[256*k + i] = tanh(x);
}
__global__ void Fprop2(const bool* drop, const float* in, const float* syn2, float* layer2)
{
   int i = threadIdx.x;                         //256
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //256
   int k = blockIdx.x;                          //Data.count

   if (drop[i])
   {
      float x = 0.0;
      for (int j=0; j < 256; ++j)
      {
         if (drop[j])
            x += in[256*k + j] * syn2[j*256 + i];
      }
      layer2[256*k + i] = x;
   }
}
__global__ void LSTM2(const bool* drop, const float* layer2i, float* layer2o, float* lstm2, const float* gate2i, const float* gate2o, float* dgate2o, const int offset)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //256

   if (drop[i])
   {
      float g_i = sigmoid(gate2i[256*offset + i]);
      float g_f = 1.0 - g_i;
      float g_o = sigmoid(gate2o[256*offset + i]);
      dgate2o[256*offset + i] = g_o;

      float i_c = g_i * tanh(layer2i[256*offset + i]);
      float i_p = 0.0;
      if (offset > 0)
         i_p = g_f * lstm2[256*(offset-1) + i];
      float sum = i_p + i_c;
      lstm2[256*offset + i] = sum;
      layer2o[256*offset + i] = tanh(sum) * g_o;
   }
}
__global__ void FpropH2(const bool* drop, float* layer2, const float* layer2o, const float* synH2, const int offset)
{
   int i = blockIdx.x; //256
   int j = threadIdx.x; //256

//   float x = 0.0;
//#pragma unroll
//   for (int i=0; i < 256; ++i)
//      x +=  layer2o[256*(offset-1) + i] * synH2 [i*256 + j];
//   layer2[256*offset + j] = x;
   if (drop[i] && drop[j])
      atomicAdd(&layer2[256*offset + j] , layer2o[256*(offset-1) + i] * synH2 [i*256 + j]);
}
__global__ void Fprop3(const bool* drop, const float* layer2, const float* syn3, float* out)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //4
   int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count
   //int k = blockDim.y*blockIdx.y + threadIdx.y; //256
   //atomicAdd(&out[j*4 + i], layer2[256*j + k] * syn3[k*4 + i]);

   float x = 0.0;
#pragma unroll
   for (int k=0; k < 256; ++k)
   {
      if (drop[i])
         x += layer2[256*j + k] * syn3[k*4 + i];
   }
   out[j*4 + i] = sigmoid(x);
}
__global__ void Ecalc3(float* out, const float* label)
{
   int i = threadIdx.x;    //4
   int j = blockIdx.x;     //Data.count

   out[i] = label[i] - out[4*j + i];
}
__global__ void Dcalc3(float* out, const float* label)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //4
   int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count

   float x = label[i] - out[4*j + i];
   //out[4*j + i] = x * dsigmoid(out[4*j + i]);
   out[4*j + i] = x;
}
__global__ void Bprop3(const bool* drop, const float* out, const float* layer2, float* dsyn3, const float alpha)
{
   int i = blockDim.y*blockIdx.y + threadIdx.y; //256
   int j = threadIdx.x; //4
   int k = blockIdx.x;  //Data.count

   if (drop[i])
      atomicAdd(&dsyn3[i*4 + j], out[4*k + j] * layer2[256*k + i] * -alpha);
}
__global__ void Dcalc2(const bool* drop, const float* out, float* dlayer2, const float* syn3)
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
         x += out[j*4 + k] * syn3[i*4 + k];
         y += out[l*4 + k] * syn3[i*4 + k];
      }
      dlayer2[j*256 + i] = (x+y)/2;
   }
}
__global__ void BpropH2(const bool* drop, const float* dlstm2, const float* gate2i, const float* gate2o, const float* dgate2o, const float* lstm2, float* dsynH2, float* dsynH2i, float* dsynH2o, const float alpha)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //256
   int offset = blockIdx.y+1; //Data.count-1

   if (drop[i])
   {
      float d_o = tanh(lstm2[(offset-1)*256 + i]) * dgate2o[(offset-1)*256 + i];
      atomicAdd(&dsynH2 [i*256 + j] , dlstm2[offset*256 + j] * d_o * -alpha);
      atomicAdd(&dsynH2i[i*256 + j] , gate2i[offset*256 + j] * d_o * -alpha);
      atomicAdd(&dsynH2o[i*256 + j] , gate2o[offset*256 + j] * d_o * -alpha);
   }
}
__global__ void DcalcH2(const bool* drop, const float* layer2i, float* dlayer2, const float* synH2, const int offset)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //256

   if (drop[i] && drop[j])
   {
      //float x = 0.0;
      //for (int j=0; j < 256; ++j)
      //   x += layer2i[offset*256 + j] * synH2[i*256 + j];
      //dlayer2[(offset-1)*256 + i] += x;
      atomicAdd(&dlayer2[(offset-1)*256 + i] , layer2i[offset*256 + j] * synH2[i*256 + j]);
   }
}
__global__ void BLSTM2(const bool* drop, const float* layer2i, const float* layer2o, const float* dlayer2, const float* lstm2, float* gate2i, float* gate2o)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //Data.count

   if (drop[i])
   {
      float d_o = tanh(lstm2[256*j + i])   * dlayer2[j*256 + i];
      float d_i = tanh(layer2i[256*j + i]) * layer2o[j*256 + i];
      float d_f = 0.0;
      if (j != 0)
         d_f = lstm2[256*(j-1) + i] * layer2o[j*256 + i];;
      //gate2o[256*j + i] = d_o * dsigmoid(sigmoid(gate2o[j*256 + i]));
      //gate2i[256*j + i] = (d_i - d_f) * dsigmoid(sigmoid(gate2i[256*j + i]));
      gate2o[256*j + i] = d_o;
      gate2i[256*j + i] = (d_i - d_f);
   }
}
__global__ void BLSTMH2(const bool* drop, const float* layer2i, float* layer2o, const float* dlayer2, float* dlstm2, const float* lstm2, const float* gate2i, const float* gate2o, const int offset, const bool last)
{
   int i = threadIdx.x; //256
   int j = offset;

   if (drop[i])
   {
      float e_c = dlayer2[j*256 + i];

      float e_s = sigmoid(gate2o[256*j + i]) * dtanh(layer2o[256*j + i]) * e_c;
      //float e_s = sigmoid(gate2o[256*j + i]) * min(e_c*e_c, 1.0) * e_c;
      if (!last)
         e_s += (1.0 - sigmoid(gate2i[256*(j+1) + i])) * layer2o[256*(j+1) + i];
      layer2o[256*j + i] = e_s;
      float d_c = sigmoid(gate2i[256*j + i]) * dtanh(layer2i[256*j + i]) * e_s;
      //float d_c = sigmoid(gate2i[256*j + i]) * min(e_s*e_s, 1.0) * e_s;
      dlstm2 [256*j + i] = d_c;
   }
}
__global__ void Bprop2(const bool* drop, const float* dlstm2, const float* gate2i, const float* gate2o, const float* in, float* dsyn2, float* dsyn2i, float* dsyn2o, const float alpha)
{
   //int i = blockDim.y*blockIdx.y + threadIdx.y; //256
   int j = threadIdx.x;                         //256
   int k = blockIdx.x;                          //Data.count

   if (drop[j])
   {
      for (int i=0; i < 256; ++i)
      {
         if (dlstm2[k*256 + j] != 0.0)
            atomicAdd(&dsyn2[i*256 + j]  , dlstm2[k*256 + j] * in[k*256 + i] * -alpha);
         if (gate2i[k*256 + j] != 0.0)
            atomicAdd(&dsyn2i[i*256 + j] , gate2i[k*256 + j] * in[k*256 + i] * -alpha);
         if (gate2o[k*256 + j] != 0.0)
            atomicAdd(&dsyn2o[i*256 + j] , gate2o[k*256 + j] * in[k*256 + i] * -alpha);
      }
   }
}
__global__ void Dcalc1(const float* dlayer2, float* layer1, const float* syn2)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x; //Data.count

   float x = 0.0;
//#pragma unroll
   for (int k=0; k < 256; ++k)
   {
      x += dlayer2[j*256 + k] * syn2[i*256 + k];
   }
   layer1[j*256 + i] = x;
}
__global__ void Bprop1(const float* layer1, const float* in, float* dsyn1, const float alpha)
{
   //int i = blockDim.y*blockIdx.y + threadIdx.y; //64
   int j = threadIdx.x;                         //256
   int k = blockIdx.x;                          //Data.count

   for (int i=0; i < 64; ++i)
   {
      atomicAdd(&dsyn1[i*256 + j]  , layer1[k*256 + j] * in[k*64 + i] * -alpha);
   }
}


void pickle(float* syn2, float* syn2i, float* syn2o, float* synH2, float* synH2i, float* synH2o, float* syn3, char* filename)
{
   int er = 0;
   FILE* outfile = fopen(filename, "wb");
   er += fwrite(syn2, sizeof(float), 64*256, outfile);
   er += fwrite(syn2i,sizeof(float), 64*256, outfile);
   er += fwrite(syn2i,sizeof(float), 64*256, outfile);
   er += fwrite(synH2, sizeof(float), 256*256, outfile);
   er += fwrite(synH2i,sizeof(float), 256*256, outfile);
   er += fwrite(synH2o,sizeof(float), 256*256, outfile);
   er += fwrite(syn3, sizeof(float), 256*4, outfile);
   printf("%d\n", er);
   fclose(outfile);
}
void unpickle(float* syn2, float* syn2i, float* syn2o, float* synH2, float* synH2i, float* synH2o, float* syn3, char* filename)
{
   int er = 0;
   FILE* infile = fopen(filename, "rb");
   er += fread(syn2, sizeof(float), 64*256, infile);
   er += fread(syn2i,sizeof(float), 64*256, infile);
   er += fread(syn2o,sizeof(float), 64*256, infile);
   er += fread(synH2, sizeof(float), 256*256, infile);
   er += fread(synH2i,sizeof(float), 256*256, infile);
   er += fread(synH2o,sizeof(float), 256*256, infile);
   er += fread(syn3, sizeof(float), 256*4, infile);
   printf("%d\n", er);
   fclose(infile);
}

void search(float* layer, int n)
{
   float min=INFINITY, max=0;
   bool nans = false;
   for (int i=0; i < n; ++i)
   {
      if (isnan(layer[i]))
         nans = true;
      else
      {
         if (abs(layer[i]) > abs(max))
            max = layer[i];
         if (abs(layer[i]) < abs(min) && abs(layer[i]) != 0.0)
            min = layer[i];
      }
   } 
   if (nans)
      printf("\tnans present\n");
   printf("\tmin: %f\n\tmax: %f\n", min, max);
}


int main(int argc, char** argv)
{
   int num_data = 0;;
   int num_test = 0;

   struct data* Data = NULL;
   struct data* Test = NULL;

   //float* d_in;     err(cudaMalloc((void**)&d_in,     64*Data.count*sizeof(float)));
   //float* d_label;  err(cudaMalloc((void**)&d_label,  Data.count*4*sizeof(float)));
   //float* d_layer2; err(cudaMalloc((void**)&d_layer2, Data.count*256*sizeof(float)));
   //float* d_dlayer2;err(cudaMalloc((void**)&d_dlayer2,Data.count*256*sizeof(float)));
   //float* d_out;    err(cudaMalloc((void**)&d_out,    Data.count*4*sizeof(float)));
   float* d_syn1=NULL;   err(cudaMalloc((void**)&d_syn1,   64*256*sizeof(float)));
   float* d_dsyn1=NULL;  err(cudaMalloc((void**)&d_dsyn1,  64*256*sizeof(float)));
   float* d_syn2=NULL;   err(cudaMalloc((void**)&d_syn2,   256*256*sizeof(float)));
   float* d_dsyn2=NULL;  err(cudaMalloc((void**)&d_dsyn2,  256*256*sizeof(float)));
   float* d_syn2i=NULL;  err(cudaMalloc((void**)&d_syn2i,  256*256*sizeof(float)));
   float* d_dsyn2i=NULL; err(cudaMalloc((void**)&d_dsyn2i, 256*256*sizeof(float)));
   float* d_syn2o=NULL;  err(cudaMalloc((void**)&d_syn2o,  256*256*sizeof(float)));
   float* d_dsyn2o=NULL; err(cudaMalloc((void**)&d_dsyn2o, 256*256*sizeof(float)));
   float* d_synH2=NULL;   err(cudaMalloc((void**)&d_synH2,   256*256*sizeof(float)));
   float* d_dsynH2=NULL;  err(cudaMalloc((void**)&d_dsynH2,  256*256*sizeof(float)));
   float* d_synH2i=NULL;  err(cudaMalloc((void**)&d_synH2i,  256*256*sizeof(float)));
   float* d_dsynH2i=NULL; err(cudaMalloc((void**)&d_dsynH2i, 256*256*sizeof(float)));
   float* d_synH2o=NULL;  err(cudaMalloc((void**)&d_synH2o,  256*256*sizeof(float)));
   float* d_dsynH2o=NULL; err(cudaMalloc((void**)&d_dsynH2o, 256*256*sizeof(float)));
   float* d_syn3=NULL;   err(cudaMalloc((void**)&d_syn3,   256*4*sizeof(float)));
   float* d_dsyn3=NULL;  err(cudaMalloc((void**)&d_dsyn3,  256*4*sizeof(float)));
   float* d_label=NULL;  err(cudaMalloc((void**)&d_label,  4*sizeof(float)));

   float weights1[64*256];    //input to middle layer weights
   float weights2[256*256];     //middle to middle layer weights
   float weights2i[256*256];    //middle to middle layer weights
   float weights2o[256*256];    //middle to middle layer weights
   float weightsH2[256*256];   //propagation through time weights
   float weightsH2i[256*256];   //propagation through time weights
   float weightsH2o[256*256];   //propagation through time weights
   float weights3[256*4];     //middle to output layer weights

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
      unpickle(weights2, weights2i, weights2o, weightsH2, weightsH2i, weightsH2o, weights3, argv[argc-2]);
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
            float r = 1.0/sqrt(64);
            weights1  [j*256 + k] = (float)rand()/RAND_MAX*2*r - r;
         }
      }
      for (int i=0; i < 256; ++i)
      {
         for (int j=0; j < 4; ++j)
         {
            float r = 1.0/sqrt(256);
            weights3[i*4 + j] = (float)rand()/RAND_MAX*2*r - r;
         }
      }

      for (int i=0; i < 256; ++i)
      {
         for (int j=0; j < 256; ++j)
         {
            float r = 1.0/sqrt(256);
            weights2  [i*256 + j] = (float)rand()/RAND_MAX*2*r - r;
            weights2i [i*256 + j] = (float)rand()/RAND_MAX*2*r - r;
            weights2o [i*256 + j] = (float)rand()/RAND_MAX*2*r - r;
            weightsH2 [i*256 + j] = (float)rand()/RAND_MAX*2*r - r;
            weightsH2i[i*256 + j] = (float)rand()/RAND_MAX*2*r - r;
            weightsH2o[i*256 + j] = (float)rand()/RAND_MAX*2*r - r;
         }
      }


      //err(cudaMemcpy(d_in, Data.Image, 64*Data.count*sizeof(float), cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_label, Data.Label, 4*Data.count*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn1,  weights1, 64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn1, weights1, 64*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn2,  weights2, 256*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn2, weights2, 256*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn2i, weights2i,256*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn2i,weights2i,256*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn2o, weights2o,256*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn2o,weights2o,256*256*sizeof(float), cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_synH2,  weightsH2, 256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsynH2, weightsH2, 256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_synH2i, weightsH2i,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsynH2i,weightsH2i,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_synH2o, weightsH2o,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsynH2o,weightsH2o,256*256*sizeof(float),cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_syn3,  weights3, 4*256*sizeof(float),  cudaMemcpyHostToDevice));
      err(cudaMemcpy(d_dsyn3, weights3, 4*256*sizeof(float),  cudaMemcpyHostToDevice));

      //train
      float alpha = atof(argv[num_args-1])/num_data;
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
            float* d_layer1;  float* layer1;
            float* d_layer2i; float* layer2i;
            float* d_layer2o; float* layer2o;
            float* d_gate2i;  float* gate2i;
            float* d_gate2o;  float* gate2o;
            float* d_dgate2o; float* dgate2o;
            float* d_dlayer2; float* dlayer2;
            float* d_dlstm2;  float* dlstm2;
            float* d_lstm2;   float* lstm2;
            float* d_out;     float* out;

            err(cudaMalloc((void**)&d_in,     Data[d].count*64*sizeof(float)));
            err(cudaMalloc((void**)&d_layer1, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_layer2i,Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_layer2o,Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_gate2i, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_gate2o, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_dgate2o, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_dlayer2,Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_dlstm2, Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_lstm2,  Data[d].count*256*sizeof(float)));
            err(cudaMalloc((void**)&d_out,    Data[d].count*4*sizeof(float)));
            layer1  = (float*)malloc(Data[d].count*256*sizeof(float));
            layer2i = (float*)malloc(Data[d].count*256*sizeof(float));
            layer2o = (float*)malloc(Data[d].count*256*sizeof(float));
            gate2i  = (float*)malloc(Data[d].count*256*sizeof(float));
            gate2o  = (float*)malloc(Data[d].count*256*sizeof(float));
            dgate2o = (float*)malloc(Data[d].count*256*sizeof(float));
            dlayer2 = (float*)malloc(Data[d].count*256*sizeof(float));
            dlstm2  = (float*)malloc(Data[d].count*256*sizeof(float));
            lstm2   = (float*)malloc(Data[d].count*256*sizeof(float));
            out     = (float*)malloc(Data[d].count*4*sizeof(float));

            //for (int i=0; i < Data[d].count*256; ++i)
            //   layer2[i] = 0;
            //for (int i=0; i < 4; ++i)
            //   out[i] = 0;

            //cudaDeviceSynchronize();
            err(cudaMemset(d_layer1,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_layer2i, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_layer2o, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_gate2i,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_gate2o,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_dgate2o, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_dlayer2, 0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_dlstm2,  0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_lstm2,   0.0, Data[d].count*256*sizeof(float)));
            err(cudaMemset(d_out,     0.0, Data[d].count*4  *sizeof(float)));

            err(cudaMemcpy(d_in,    Data[d].Image, 64*Data[d].count*sizeof(float), cudaMemcpyHostToDevice));
            err(cudaMemcpy(d_label, Data[d].Label, 4*sizeof(float), cudaMemcpyHostToDevice));

            //printf("input:\n");
            //search(Data[d].Image, Data[d].count*64);

      // forward pass
            Fprop1<<<Data[d].count, 256>>>(d_in, d_syn1, d_layer1);
            Fprop2<<<Data[d].count, 256 ,0, s[0]>>>(d_drop, d_layer1, d_syn2, d_layer2i);
            Fprop2<<<Data[d].count, 256 ,0, s[1]>>>(d_drop, d_layer1, d_syn2i, d_gate2i);
            Fprop2<<<Data[d].count, 256 ,0, s[2]>>>(d_drop, d_layer1, d_syn2o, d_gate2o);
            err(cudaMemcpy(layer2i,d_layer2i,Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            printf("layer2i:\n");
            search(layer2i,Data[d].count*256);
            LSTM2<<<1, 256>>>(d_drop, d_layer2i, d_layer2o, d_lstm2, d_gate2i, d_gate2o, d_dgate2o, 0);
            for (int i=1; i < Data[d].count; ++i)
            {
               FpropH2<<<256, 256 ,0, s[0]>>>(d_drop, d_layer2i, d_layer2o, d_synH2,  i);
               FpropH2<<<256, 256 ,0, s[1]>>>(d_drop, d_gate2i,  d_layer2o, d_synH2i, i);
               FpropH2<<<256, 256 ,0, s[2]>>>(d_drop, d_gate2o,  d_layer2o, d_synH2o, i);
               LSTM2<<<1, 256>>>(d_drop, d_layer2i, d_layer2o, d_lstm2, d_gate2i, d_gate2o, d_dgate2o, i);
            }
            Fprop3<<<dim3(1, Data[d].count), dim3(4, 1)>>>(d_drop, d_layer2o, d_syn3, d_out);

            err(cudaMemcpy(d_dsyn3, weights3, 4*256*sizeof(float),  cudaMemcpyHostToDevice));
            err(cudaMemcpy(layer1, d_layer1, Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(layer2i,d_layer2i,Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(layer2o,d_layer2o,Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(gate2i, d_gate2i, Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(gate2o, d_gate2o, Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(dgate2o,d_dgate2o,Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(dlayer2,d_dlayer2,Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(dlstm2, d_dlstm2, Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(lstm2,  d_lstm2,  Data[d].count*256*sizeof(float), cudaMemcpyDeviceToHost));
            err(cudaMemcpy(out,    d_out,    Data[d].count*4*sizeof(float)  , cudaMemcpyDeviceToHost));

            //printf("layer1:\n");
            //search(layer1, Data[d].count*256);
            //printf("layer2i:\n");
            //search(layer2i,Data[d].count*256);
            //printf("layer2o:\n");
            //search(layer2o,Data[d].count*256);
            //printf("gate2i:\n");
            //search(gate2i, Data[d].count*256);
            //printf("gate2o:\n");
            //search(gate2o, Data[d].count*256);
            //printf("dgate2o:\n");
            //search(dgate2o,Data[d].count*256);
            //printf("dlayer2:\n");
            //search(dlayer2,Data[d].count*256);
            //printf("dlstm2:\n");
            //search(dlstm2, Data[d].count*256);
            //printf("lstm2:\n");
            //search(lstm2,  Data[d].count*256);
            //printf("out:\n");
            //search(out,    Data[d].count*4);

      // backward pass
            Dcalc3<<<dim3(1, Data[d].count), dim3(4, 1)>>>(d_out, d_label);
            Bprop3<<<dim3(Data[d].count,1), dim3(4,256)>>>(d_drop, d_out, d_layer2o, d_dsyn3, alpha/Data[d].count);
            Dcalc2<<<Data[d].count, 256>>>(d_drop, d_out, d_dlayer2, d_syn3);
            bool last = true;
            for (int i=Data[d].count-1; i >= 1; i -= 1)
            {
               BLSTMH2<<<1, 256>>>(d_drop, d_layer2i, d_layer2o, d_dlayer2, d_dlstm2, d_lstm2, d_gate2i, d_gate2o, i, last);
               DcalcH2<<<256, 256>>>(d_drop, d_layer2i, d_dlayer2, d_synH2, i);

               //err(cudaMemcpy(layer2, d_dlstm2, sizeof(float)*Data[d].count*256, cudaMemcpyDeviceToHost));
               //bool thing = false;
               //for (int j=0; j < Data[d].count*256; ++j)
               //{
               //   if (isnan(layer2[j]))
               //      thing = true;
               //}
               //if (thing)
               //{
               //   printf("problem at i = %d/%d\n", i, Data[d].count-1);
               //   break;
               //}
               last = false;
            }
            BLSTMH2<<<1, 256>>>(d_drop, d_layer2i, d_layer2o, d_dlayer2, d_dlstm2, d_lstm2, d_gate2i, d_gate2o, 0, false);

            BLSTM2<<<Data[d].count, 256>>>(d_drop, d_layer2i, d_layer2o, d_dlayer2, d_lstm2, d_layer2i, d_layer2o);
            BpropH2<<<dim3(256, Data[d].count-2), dim3(256, 1) ,0, s[0]>>>(d_drop, d_dlstm2, d_gate2i, d_gate2o, d_dgate2o, d_lstm2, d_dsynH2, d_dsynH2i, d_dsynH2o, alpha/Data[d].count);
            Bprop2<<<Data[d].count, 256 ,0, s[1]>>>(d_drop, d_dlstm2, d_gate2i, d_gate2o, d_layer1, d_dsyn2, d_dsyn2i, d_dsyn2o, alpha/Data[d].count);
            Dcalc1<<<Data[d].count, 256>>>(d_dlayer2, d_syn2, d_layer1);
            Bprop1<<<Data[d].count, 256>>>(d_layer1, d_in, d_dsyn1, alpha/Data[d].count);

            err(cudaFree(d_in));
            err(cudaFree(d_layer1));
            err(cudaFree(d_layer2i));
            err(cudaFree(d_layer2o));
            err(cudaFree(d_gate2i));
            err(cudaFree(d_gate2o));
            err(cudaFree(d_dgate2o));
            err(cudaFree(d_dlayer2));
            err(cudaFree(d_lstm2));
            err(cudaFree(d_dlstm2));
            err(cudaFree(d_out));

            cudaDeviceSynchronize();
         }

         //printf("\n");

         err(cudaMemcpy(d_syn2,  d_dsyn2, sizeof(float)*64*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_syn2i, d_dsyn2i,sizeof(float)*64*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_syn2o, d_dsyn2o,sizeof(float)*64*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_synH2,  d_dsynH2, sizeof(float)*256*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_synH2i, d_dsynH2i,sizeof(float)*256*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_synH2o, d_dsynH2o,sizeof(float)*256*256,  cudaMemcpyDeviceToDevice));
         err(cudaMemcpy(d_syn3,  d_dsyn3, sizeof(float)*256*4,   cudaMemcpyDeviceToDevice));

         err(cudaMemcpy(weights2,  d_dsyn2, sizeof(float)*64*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weights2i, d_dsyn2i,sizeof(float)*64*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weights2o, d_dsyn2o,sizeof(float)*64*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weightsH2,  d_dsynH2, sizeof(float)*256*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weightsH2i, d_dsynH2i,sizeof(float)*256*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weightsH2o, d_dsynH2o,sizeof(float)*256*256,  cudaMemcpyDeviceToHost));
         err(cudaMemcpy(weights3,  d_dsyn3, sizeof(float)*256*4,   cudaMemcpyDeviceToHost));

         //int er = 0;
         //for (int i=0; i < 64*256; ++i)
         //{
         //   if (isnan(weights2[i]))
         //      er |= 1;
         //   if (isnan(weights2i[i]))
         //      er |= 2;
         //   if (isnan(weights2o[i]))
         //      er |= 4;
         //}
         //for (int i=0; i < 256*256; ++i)
         //{
         //   if (isnan(weightsH2[i]))
         //      er |= 8;
         //   if (isnan(weightsH2i[i]))
         //      er |= 16;
         //   if (isnan(weightsH2o[i]))
         //      er |= 32;
         //}
         //for (int i=0; i < 256*4; ++i)
         //{
         //   if (isnan(weights3[i]))
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
      float* d_layer1;  //float* layer2;
      float* d_layer2i; //float* layer2i;
      float* d_layer2o; //float* layer2o;
      float* d_gate2i;  //float* gate2i;
      float* d_gate2o;  //float* gate2o;
      float* d_dgate2o; //float* gate2o;
      //float* d_dlayer2; //float* dlayer2;
      //float* d_dlstm2;  //float* dlstm2;
      float* d_lstm2;   //float* lstm2;
      float* d_out;     //float* out;

      err(cudaMalloc((void**)&d_in,     Test[d].count*64*sizeof(float)));
      err(cudaMalloc((void**)&d_layer1, Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_layer2i,Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_layer2o,Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_gate2i, Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_gate2o, Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_dgate2o,Test[d].count*256*sizeof(float)));
      //err(cudaMalloc((void**)&d_dlayer2,Test[d].count*256*sizeof(float)));
      //err(cudaMalloc((void**)&d_dlstm2, Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_lstm2,  Test[d].count*256*sizeof(float)));
      err(cudaMalloc((void**)&d_out,    Test[d].count*4*sizeof(float)));

      err(cudaMemset(d_layer1,  0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_layer2i, 0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_layer2o, 0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_gate2i,  0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_gate2o,  0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_dgate2o, 0.0, Test[d].count*256*sizeof(float)));
      //err(cudaMemset(d_dlayer2, 0.0, Test[d].count*256*sizeof(float)));
      //err(cudaMemset(d_dlstm2,  0.0, Test[d].count*256*sizeof(float)));
      err(cudaMemset(d_lstm2,   0.0, Test[d].count*256*sizeof(float)));
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

      Fprop1<<<Test[d].count, 256>>>(d_in, d_syn1, d_layer1);
      Fprop2<<<Test[d].count, 256 ,0, s[0]>>>(d_drop, d_layer1, d_syn2, d_layer2i);
      Fprop2<<<Test[d].count, 256 ,0, s[1]>>>(d_drop, d_layer1, d_syn2i, d_gate2i);
      Fprop2<<<Test[d].count, 256 ,0, s[2]>>>(d_drop, d_layer1, d_syn2o, d_gate2o);
      LSTM2<<<1, 256>>>(d_drop, d_layer2i, d_layer2o, d_lstm2, d_gate2i, d_gate2o, d_dgate2o, 0);
      for (int i=1; i < Test[d].count; ++i)
      {
         FpropH2<<<256, 256 ,0, s[0]>>>(d_drop, d_layer2i, d_layer2o, d_synH2,  i);
         FpropH2<<<256, 256 ,0, s[1]>>>(d_drop, d_gate2i,  d_layer2o, d_synH2i, i);
         FpropH2<<<256, 256 ,0, s[2]>>>(d_drop, d_gate2o,  d_layer2o, d_synH2o, i);
         LSTM2<<<1, 256>>>(d_drop, d_layer2i, d_layer2o, d_lstm2, d_gate2i, d_gate2o, d_dgate2o, i);
      }
      Fprop3<<<dim3(1, Test[d].count), dim3(4, 1)>>>(d_drop, d_layer2o, d_syn3, d_out);

      float* out = (float*)malloc(4*sizeof(float));

      err(cudaMemcpy(out, &d_out[(Test[d].count-1)*4], sizeof(float)*4,  cudaMemcpyDeviceToHost));


      err(cudaFree(d_in));
      err(cudaFree(d_layer1));
      err(cudaFree(d_layer2i));
      err(cudaFree(d_layer2o));
      err(cudaFree(d_gate2i));
      err(cudaFree(d_gate2o));
      err(cudaFree(d_dgate2o));
      err(cudaFree(d_lstm2));
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
   //      printf("%f ", weights3[i*4 + j]);
   //   }
   //   printf("\n");
   //}
   //for (int i=0; i < Test.count; ++i)
   //{

   //   //reset layer states
   //   for (int j=0; j < 256; ++j)
   //      layer2[j] = 0.0;
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
   //            layer2[l] += Test.Image[i*64 + j*28 + k] * weights2[j*28*256 + k*256 + l];
   //         }
   //      }
   //   }
   //   for (int j=0; j < 256; ++j)
   //      layer2[j] = sigmoid(layer2[j]);

   //   //middle to output layer
   //   for (int j=0; j < 256; ++j)
   //   {
   //      for (int k=0; k < 4; ++k)
   //      {
   //         outs[k] += layer2[j] * weights3[j*4 + k];
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
   //err(cudaFree(d_layer2));
   //err(cudaFree(d_dlayer2));
   err(cudaFree(d_syn1));
   err(cudaFree(d_syn2));
   err(cudaFree(d_syn2i));
   err(cudaFree(d_syn2o));
   err(cudaFree(d_synH2));
   err(cudaFree(d_synH2i));
   err(cudaFree(d_synH2o));
   err(cudaFree(d_syn3));
   err(cudaFree(d_dsyn1));
   err(cudaFree(d_dsyn2));
   err(cudaFree(d_dsyn2i));
   err(cudaFree(d_dsyn2o));
   err(cudaFree(d_dsynH2));
   err(cudaFree(d_dsynH2i));
   err(cudaFree(d_dsynH2o));
   err(cudaFree(d_dsyn3));

   if (!strcmp(argv[argc-1], "w"))
   {
      printf("writing to %s\n", argv[argc-2]);
      pickle(weights2, weights2i, weights2o, weightsH2, weightsH2i, weightsH2o, weights3, argv[argc-2]);
   }

   return EXIT_SUCCESS;
}
