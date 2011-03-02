NVCC := nvcc
NVCCFLAGS := -arch sm_13

SOURCES = hertz.cu

all: build test

build: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $^ -o aos

test: build
	./aos step1000.in step1000.out

clean:
	rm -f aos 
