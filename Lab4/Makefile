all: clean cuda1 serial1 cuda2 serial2 cuda2-print cuda3 serial3

cuda1:
	nvcc -O3 -D CUDA main.cu -o lab4p1

serial1:
	nvcc -O3 main.cu -o serial1

cuda2:
	nvcc -O3 -D CUDA main2.cu -o lab4p2

cuda2-print:
	nvcc -O3 -D CUDA -D PRINT main2.cu -o cuda2p

serial2:
	nvcc -O3 main2.cu -o serial2

cuda3:
	nvcc -O3 -D CUDA main3.cu -o lab4p3

serial3:
	nvcc -O3 main3.cu -o serial3

clean:
	rm -f lab4p1 serial1 lab4p2 serial2 cuda2p lab4p3 serial3
