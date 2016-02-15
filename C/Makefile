CFLG=-g -O3 -Wall
LIBS=-lm
CLEAN=rm -rf run *.o *.a *.dSYM

all:run

neuralnet.o:neuralnet.c reader.c
	gcc -std=c99 -c $(CFLG) $<

#  link
run:neuralnet.o
	gcc -std=c99 -g -O3 -o run $^ $(LIBS)

#  Clean
clean:
	$(CLEAN)
