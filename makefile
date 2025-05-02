#
# Simple makefile for coursework 3.
#
# Should work on both a School machine once CUDA has been loaded (see Lecture 14),
# and also on a Mac. Note however that all submissions will be assessed on a School
# machine similar to the remote GPU-enabled machine feng-linux.leeds.ac.uk/gpu.
#
EXE = cwk3
OS = $(shell uname)

ifeq ($(OS), Linux)
	MSG = On a School machine, first load a suitable CUDA module; see Lecture 14 for details.
	MSG += If it does not work, you can try another machine in Bragg 2.05, or the GPU-enabled
	MSG += remote machine feng-linux.leeds.ac.uk/gpu (don't forget the 'gpu'!)
	CC = nvcc -lOpenCL
endif

ifeq ($(OS), Darwin)
	MSG = Using the clang compiler. May need to change compiler and options depending on your set-up.
	CC = clang -framework OpenCL
endif

all:
	@echo "$(MSG)"
	@echo
	$(CC) -o $(EXE) cwk3.c



