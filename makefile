CC = gcc
CFLAGS = -Wall -Werror -fsanitize=address  -g

.PHONY: clean default learn

default: learn
learn: learn.c
	$(CC) $(CFLAGS) -o learn learn.c

clean:
	rm -f *.o learn
