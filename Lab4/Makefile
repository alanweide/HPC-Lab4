all: cuda serial

cuda:
	nvcc -O3 -D CUDA main.cu -o cuda

serial:
	nvcc -O3 main.cu -o serial