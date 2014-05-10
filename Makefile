CC=gcc
#INCLUDE=
CFLAGS=-Wall -fPIC -O3 -std=c99 -fopenmp

all: libfastsimilarity.so

temp:  
	mkdir temp

temp/fastsimilarity.o: temp
	$(CC) -c similarity.c -o $@ ${CFLAGS} ${INCLUDE}

libfastsimilarity.so: temp/fastsimilarity.o 
	$(CC) -shared -o $@ temp/*.o ${INCLUDE}

clean:
	rm -rf temp
	rm libfastsimilarity.so
