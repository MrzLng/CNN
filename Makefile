# Variables
CXX = g++
CXXFLAGS = -std=c++17 -I./include -I./external/stb
TARGET = bin/cnn_app.exe
SRCS = src/main.cpp src/image.cpp src/convolution.cpp src/buffer.cpp

all: create_bin $(TARGET)
	@echo ---------------------------------------
	@echo Build complete: $(TARGET)
	@echo Executing: $(TARGET)
	@echo ---------------------------------------
	@$(TARGET)


create_bin:
	@if not exist bin mkdir bin

$(TARGET): $(SRCS)
	@echo Compiling source files...
	@$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) || (echo [ERROR] Compilation failed! Check your code syntax. && exit 1)

run:
	$(TARGET)

clean:
	@if exist bin rd /s /q bin
	@echo [CLEAN] Workspace is now empty.