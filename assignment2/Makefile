NVCC = nvcc
NVCCFLAGS = -O2 -std=c++20 -ccbin g++-14
LDFLAGS = -lcusparse

TARGET = out/wave2d
SRC = wave2d.cu

all: $(TARGET)

$(TARGET): $(SRC) | out
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

out:
	mkdir -p out

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf out

.PHONY: all run clean
