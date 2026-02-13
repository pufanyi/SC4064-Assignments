NVCC = nvcc
NVCCFLAGS = -O2 -std=c++20 -ccbin g++-14

TARGET = out/vector_add
SRC = vector_add.cu

all: $(TARGET)

$(TARGET): $(SRC) | out
	$(NVCC) $(NVCCFLAGS) -o $@ $<

out:
	mkdir -p out

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
