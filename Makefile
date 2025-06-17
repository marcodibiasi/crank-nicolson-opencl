TARGET = Project
SRCS = main.c pgm_utils.c solver.c
OBJS = $(SRCS:.c=.o)
CC = gcc

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)