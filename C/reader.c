#include <stdio.h>
#include <stdlib.h>

int bitswap(int x)  //necessary for the MNIST datset when run on Intel processors
{
   int y = 0;
   y |= (x & 0x000000FF) << 24;
   y |= (x & 0x0000FF00) << 8;
   y |= (x & 0x00FF0000) >> 8;
   y |= (x & 0xFF000000) >> 24;
   return y;
}

struct data
{
   float** Label;
   float*** Image;
   int count, height, width;
};

struct data read(char* images, char* labels)
{
   struct data Result;
   Result.Label = NULL;
   Result.Image = NULL;

   int magic;
   int imagic;
   int count;
   int imcount;
   int width;
   int height;
   FILE* labelfile = fopen(labels, "rb");
   if (!labelfile)
   {
      printf("missing file\n");
      return Result;
   }

   fread(&magic, 4,1, labelfile); magic = bitswap(magic);
   fread(&count, 4,1, labelfile); count = bitswap(count);
   printf("%d,%d\n", magic, count);
   if (magic != 2049)
   {
      printf("file format error: %d\n", magic);
      fclose(labelfile);
      return Result;
   }
   unsigned char *label = malloc(sizeof(unsigned char)*count);
   float **Label = malloc(sizeof(float*)*count);
   for (int i=0; i<count; ++i)
   {
      fread(&label[i], 1,1, labelfile);
      Label[i] = malloc(sizeof(float)*10);
      for (int j=0; j < 10; ++j)
      {
         if (label[i] == j)
            Label[i][j] = 1.0;
         else
            Label[i][j] = 0.0;
      }
   }
   fclose(labelfile);
   free(label);

   FILE* imagefile = fopen(images, "rb");

   fread(&imagic, 4,1, imagefile); imagic = bitswap(imagic);
   fread(&imcount, 4,1, imagefile); imcount = bitswap(imcount);
   fread(&height, 4,1, imagefile); height = bitswap(height);
   fread(&width, 4,1, imagefile); width = bitswap(width);
   if (imagic != 2051 || imcount != count)
   {
      printf("file format error: %d, %d\n", imagic, imcount);
      fclose(imagefile);
      return Result;
   }
   printf("%d,%d\n", imagic, imcount);
   printf("%d,%d\n", height, width);
   printf("%d\n", imcount*height*width);

   float*** image = malloc(sizeof(float**)*imcount);
   for (int i=0; i<imcount; ++i)
   {
      image[i] = malloc(sizeof(float*)*height);
      for (int j=0; j<height; ++j)
         image[i][j] = malloc(sizeof(float)*width);
   }
   unsigned char tmp;

   for (int i=0; i<imcount; ++i)
   {
      for (int j=0; j<height; ++j)
      {
         for (int k=0; k<width; ++k)
         {
            fread(&tmp, 1,1, imagefile);
            image[i][j][k] = (float)tmp/256.0;
         }
      }
   }
   fclose(imagefile);

   Result.Label = Label;
   Result.Image = image;
   Result.count = imcount;
   Result.height = height;
   Result.width = width;
   
   return Result;
}
