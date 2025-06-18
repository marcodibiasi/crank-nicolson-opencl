TARGET = Project
SRCS = main.c pgm_utils.c solver.c
OBJS = $(SRCS:.c=.o)
CC = gcc

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    OPENCL_FLAGS = -framework OpenCL
else ifeq ($(UNAME_S),Linux)
    OPENCL_FLAGS = -lOpenCL
else ifeq ($(OS),Windows_NT)
    OPENCL_FLAGS = -lOpenCL
endif

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(OPENCL_FLAGS)