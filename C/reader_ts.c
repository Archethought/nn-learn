#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sndfile.h>
#include <fftw3.h>
#include <math.h>

void ferr(int code)
{
   if (code < 0) fprintf(stderr, "oops, read error: %d\n", code);
}

struct data
{
   float* Label;
   float* Image;
   int count, depth;
};

struct data read_this(const char* infile, const char* label)
{
   struct data Result;
   Result.Label = NULL;
   Result.Image = NULL;

   //char infile[256] = {0};
   //strcpy(infile, images);

   SF_INFO info;// = {0};
   memset(&info, 0, sizeof(info));
   //printf("%s\n", infile);
   //SNDFILE *file = sf_open("waterfall.wav", SFM_READ, &info);
   SNDFILE *file;
   file = sf_open(infile, SFM_READ, &info);
   if (file == NULL) {printf("file error: %s\n%s\n", infile, sf_strerror(file)); exit(1);}
   float* raw_data = (float*)malloc(info.frames*sizeof(float));

   //sf_seek(file, 0, SEEK_SET);
   sf_readf_float(file, raw_data, info.frames);

   Result.Label = (float*)malloc(4*sizeof(float));
   if (!strcmp(label, "classical"))
   {
      for (int i=0; i < 4; ++i)
      {
         if (i == 0)
            Result.Label[i] = 1.0;
         else
            Result.Label[i] = 0.0;
      }
   }
   else if (!strcmp(label, "jazz"))
   {
      for (int i=0; i < 4; ++i)
      {
         if (i == 1)
            Result.Label[i] = 1.0;
         else
            Result.Label[i] = 0.0;
      }
   }
   else if (!strcmp(label, "metal"))
   {
      for (int i=0; i < 4; ++i)
      {
         if (i == 2)
            Result.Label[i] = 1.0;
         else
            Result.Label[i] = 0.0;
      }
   }
   else if (!strcmp(label, "pop"))
   {
      for (int i=0; i < 4; ++i)
      {
         if (i == 3)
            Result.Label[i] = 1.0;
         else
            Result.Label[i] = 0.0;
      }
   }
   else
   {
      printf("Label not supported: %s", label);
      free(raw_data);
      free(Result.Label);
      exit(1);
   }

   //Result.Image = Result.Image;
   //Result.count = info.frames;
   //Result.depth = info.channels;

   sf_close(file);

   //printf("Info struct: \nframes:\t%d\nsamplerate:\t%d\nchannels:\t%d\nformat:\t%x\nsections:\t%d\nseekable:\t%d\n", (int)info.frames, info.samplerate, info.channels, info.format, info.sections, info.seekable);

   int segment_length = info.samplerate/50;
   int fft_depth = 64;
   Result.depth = fft_depth;
   int samples_per_bin = 5*info.samplerate/22050;

   double* in;
   fftw_complex* out;
   double* processed;
   fftw_plan plan;

   in = (double*) fftw_malloc(segment_length*sizeof(double));
   out = (fftw_complex*) fftw_malloc(segment_length*sizeof(fftw_complex));
   processed = (double*)malloc(segment_length*sizeof(double));
   plan = fftw_plan_dft_r2c_1d(segment_length, in, out, FFTW_ESTIMATE);

   Result.Image = (float*)malloc(fft_depth * (info.frames/segment_length) * sizeof(float));

   //float max = 0.0;
   for (int i=0; (i+1)*segment_length < info.frames; ++i)
   {
      //copy the next segment to in[]
      for (int j=0; j < segment_length; ++j)
      {
         in[j] = (double)raw_data[i*segment_length + j];
         out[j][0] = out[j][1] = 0;
      }

      //run the transform to out[]
      fftw_execute(plan);

      //post-processing on out[]
#pragma omp parallel for
      for (int j=0; j < segment_length; ++j)
      {
         out[j][0] *= 2.0/segment_length;
         out[j][1] *= 2.0/segment_length;
         processed[j] = fabs(powf(out[j][0]*out[j][0] + out[j][1]*out[j][1], 1.0/3.0));
         //if (fabs(processed[j]) > max) max = (float)fabs(processed[j]);
      }

      //sum up and copy to the next segment of Result.Image[]
#pragma omp parallel for
      for (int j=0; j < fft_depth; ++j)
      {
         if (i*fft_depth + j < fft_depth*(info.frames/segment_length))
         {
            for (int k=0; k<samples_per_bin; ++k)
               Result.Image[i*fft_depth + j] += (float)processed[j*samples_per_bin + k];
            Result.Image[i*fft_depth + j] /= samples_per_bin;
         }
      }
      Result.count = i;
   }
   //printf("max: %f\n", max);

   free(raw_data);
   free(processed);
   fftw_free(in);
   fftw_free(out);
   fftw_destroy_plan(plan);

   return Result;
}

//int main(int argc, char** argv)
//{
//   //printf("%s\n", argv[1]);
//   const char* input = argv[1];
//   struct data Thing = read_this(input, "pop");
//   printf("thing1: %d\n", Thing.depth);
//   printf("thing2: %d\n", Thing.count);
//   printf("thing3: %f\n", Thing.Image[Thing.depth*Thing.count]);
//   free(Thing.Image);
//   free(Thing.Label);
//   return 0;
//}
