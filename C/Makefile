CFLG=-g -O3 -Wall
LIBS=-lm
CLEAN=rm -rf *run *.o *.a *.dSYM

all:run grun srun

neuralnet.o:neuralnet.c reader.c
	gcc -std=c99 -c $(CFLG) $<

nn.o:nn.cu reader.c
	nvcc -ccbin gcc -m64 -gencode arch=compute_52,code=sm_52 -c -g -G -O3 $<
NN.o:NN.cu reader_.c
	nvcc -ccbin gcc -m64 -gencode arch=compute_52,code=sm_52 -c -g -G -O3 $<

#  link
run:neuralnet.o
	gcc -std=c99 -g -O3 -o run $^ $(LIBS)

grun:nn.o
	nvcc -ccbin gcc -m64 -gencode arch=compute_52,code=sm_52 -g -G -O3 -o $@ $^ $(GPULIBS)
srun:NN.o
	nvcc -ccbin gcc -m64 -gencode arch=compute_52,code=sm_52 -g -G -O3 -o $@ $^ $(GPULIBS)

#  Clean
clean:
	$(CLEAN)
