CFLG=-g -O3 -Wall
LIBS=-lm
CLEAN=rm -rf *run *.o *.a *.dSYM
ARCH=-gencode arch=compute_52,code=sm_52

all:run grun srun

neuralnet.o:neuralnet.c reader.c
	gcc -std=c99 -c $(CFLG) $<

NN_.o:NN_.cu reader_.c
	nvcc -ccbin gcc -m64 $(ARCH) -c -g -G -O3 $<

#  link
run:neuralnet.o
	gcc -std=c99 -g -O3 -o run $^ $(LIBS)

srun:NN_.o
	nvcc -ccbin gcc -m64 $(ARCH) -g -G -O3 -o $@ $^ $(LIBS)

#  Clean
clean:
	$(CLEAN)
