CC=gcc
#INCLUDE=
CFLAGS=-Wall -fPIC -O3 -std=c99 -fopenmp

all: libfastsimilarity.so

temp:  
	mkdir temp

temp/floatsimilarity.o: temp
	$(CC) -c floatSimilarity.c -o $@ ${CFLAGS} ${INCLUDE}

temp/intsimilarity.o: temp
	$(CC) -c  intSimilarity.c -o $@ ${CFLAGS} ${INCLUDE}

libfastsimilarity.so: temp/floatsimilarity.o temp/intsimilarity.o
	$(CC) -shared -o $@ temp/*.o ${INCLUDE}

clean:
	rm -rf temp
	rm libfastsimilarity.so
