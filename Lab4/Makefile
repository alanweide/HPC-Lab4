all: cuda serial

remake: clean all

cuda:
	nvcc -O3 -D CUDA main.cu -o cuda

long-cuda:
	nvcc -O3 -D CUDA -D REPS=10000 main.cu -o cuda

serial:
	nvcc -O3 main.cu -o serial

clean:
	rm -f cuda serial
