CC=clang
LIBS=-lm
CFLAGS=-Wall -Wextra -Wpedantic -Werror -ggdb -O3

xor: xor.c
	$(CC) $(CFLAGS) $(LIBS) xor.c -o xor

nn: nn.c
	$(CC) $(CFLAGS) $(LIBS) nn.c -o nn

mnist: mnist.c
	$(CC) $(CFLAGS) $(LIBS) mnist.c -o mnist

.PHONY: xor nn mnist

clean:
	rm -f nn
	rm -f xor
	rm -f mnist
