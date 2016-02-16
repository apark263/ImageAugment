SRCS := $(wildcard *.hpp)
CFLAGS := -Wall -Werror -O3 -std=c++11
CC := g++

loader: loader.cpp $(SRCS)
	$(CC) -o $@ $(CFLAGS) $< `pkg-config opencv --cflags --libs`
