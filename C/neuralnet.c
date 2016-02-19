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

int main(int argc, char** argv)
{
   if (argc != 7)
   {
      printf("usage: run trainingImages trainingLabels testImages testLabels iterations alpha\n");
      return 2;
   }
   struct data Data = read(argv[1], argv[2]);      //training data
   struct data Test = read(argv[3], argv[4]);      //test data

   float** input;
   float weights1[Data.height][Data.width][100];  //input to middle layer weights
   float dweights1[Data.height][Data.width][100];  //input to middle layer weights
   float layer1[100];                              //Middle layer
   float weights2[100][10];                        //middle to output layer weights
   float dweights2[100][10];                        //middle to output layer weights
   float outs[10];                                 //Output layer
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

   //train
   printf("training\n");
   int iterations = atoi(argv[5]);
   for (int iter=0; iter<iterations; ++iter)
   {
      for (int i=0; i < Data.count; ++i)
      {
         //reset layer states
         input = Data.Image[i];
         for (int j=0; j < 100; ++j)
            layer1[j] = 0.0;
         for (int j=0; j < 10; ++j)
            outs[j] = 0.0;

      // Forward pass
         //input to middle layer
         for (int j=0; j < Data.height; ++j)
         {
            for (int k=0; k < Data.width; ++k)
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
            //printf("%f ", outs[j]);
         }
         //printf("\n");
         //for (int j=0; j < 10; ++j)
         //{
         //   printf("%f ", Data.Label[i][j]);
         //}
         //printf("\n");

      //Back Propagation
      //   error[k] = labels[i][k] - outs[k]
      //   delta[k] = error * dsigmoid(outs[k])
      //   weights2[j][k] += delta[k] * layer1[j]
         //output to middle
         float delta2[10] = {0};
         for (int j=0; j < 100; ++j)
         {
            for (int k=0; k < 10; ++k)
            {
               delta2[k] = (Data.Label[i][k] - outs[k]) * dsigmoid(outs[k]);
               dweights2[j][k] += delta2[k] * layer1[j] / (Data.count/10);
            }
         }
         //middle to input
         float delta1[100] = {0};
         float error1[100] = {0};
         for (int h=0; h < Data.height; ++h)
         {
            for (int j=0; j < Data.width; ++j)
            {
               for (int k=0; k < 100; ++k)
               {
                  for (int l=0; l < 10; ++l)
                     error1[k] += delta2[l] * weights2[k][l];
                     //delta1[k] += delta2[l] * layer1[k];
                  delta1[k] = error1[k] * dsigmoid(layer1[k]);
                  dweights1[h][j][k] += delta1[k] * input[h][j] / (Data.count/10);
               }
            }
         }
      }

      //adjust synapse weights
      for (int i=0; i < Data.height; ++i)
      {
         for (int j=0; j < Data.width; ++j)
         {
            for (int k=0; k < 100; ++k)
            {
               weights1[i][j][k] += alpha * dweights1[i][j][k];
               //printf("%f ", dweights1[i][j][k]);
               dweights1[i][j][k] = 0.0;
            }
         }
      }
      //printf("\n");
      for (int i=0; i < 100; ++i)
      {
         for (int j=0; j < 10; ++j)
         {
            weights2[i][j] += alpha * dweights2[i][j];
            //printf("%f ", dweights2[i][j]);
            dweights2[i][j] = 0.0;
         }
      }
      printf("%d\n", iter);
   }

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
      input = Test.Image[i];
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
