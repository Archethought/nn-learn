CFLG=-g -O3 -Wall
LIBS=-lsndfile -lm
CLEAN=rm -rf *run Rnn *lstm* *.o *.a *.dSYM
ARCH=-gencode arch=compute_52,code=sm_52

all:run srun Rnn lstm 2lstm lstm_cpu

neuralnet.o:neuralnet.c reader.c
	gcc -std=c99 -c $(CFLG) $<
NN_.o:NN_.cu reader_.c
	nvcc -ccbin gcc -m64 $(ARCH) -c -g -G -O3 $< -D_FORCE_INLINES
RNN.o:RNN.cu reader_ts.c
	nvcc -ccbin gcc -m64 $(ARCH) -c -g -G -O3 $< -D_FORCE_INLINES -Xcompiler -fopenmp
2LSTM.o:2LSTM.cu reader_ts.c
	nvcc -ccbin gcc -m64 $(ARCH) -c -g -G -O3 $< -D_FORCE_INLINES -Xcompiler -fopenmp
LSTM.o:LSTM.cu reader_ts.c
	nvcc -ccbin gcc -m64 $(ARCH) -c -g -G -O3 $< -D_FORCE_INLINES -Xcompiler -fopenmp
LSTM_CPU.o:LSTM_CPU.c reader_ts.c
	gcc -std=c99 -c $(CFLG) $< -fopenmp

#  link
run:neuralnet.o
	gcc -std=c99 -g -O3 -o $@ $^ $(LIBS)
srun:NN_.o
	nvcc -ccbin gcc -m64 $(ARCH) -g -G -O3 -o $@ $^ $(LIBS)
Rnn:RNN.o
	nvcc -ccbin gcc -m64 $(ARCH) -g -G -O3 -o $@ $^ -lfftw3 $(LIBS) -lgomp
2lstm:2LSTM.o
	nvcc -ccbin gcc -m64 $(ARCH) -g -G -O3 -o $@ $^ -lfftw3 $(LIBS) -lgomp
lstm:LSTM.o
	nvcc -ccbin gcc -m64 $(ARCH) -g -G -O3 -o $@ $^ -lfftw3 $(LIBS) -lgomp
lstm_cpu:LSTM_CPU.o
	gcc -std=c99 -g -O3 -o $@ $^ -lfftw3 $(LIBS) -lgomp

#  Clean
clean:
	$(CLEAN)
