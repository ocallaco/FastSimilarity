CC=gcc
#INCLUDE=
CFLAGS=-Wall -fPIC -O3 -std=c99 -fopenmp

all: libfastsimilarity.so

temp:  
	mkdir temp

temp/floatsimilarity.o: temp
	$(CC) -c floatSimilarity.c -o $@ ${CFLAGS} ${INCLUDE}

<<<<<<< HEAD
libfastsimilarity.so: temp/fastsimilarity.o 
=======
temp/intsimilarity.o: temp
	$(CC) -c  intSimilarity.c -o $@ ${CFLAGS} ${INCLUDE}

libfastsimilarity.so: temp/floatsimilarity.o temp/intsimilarity.o
>>>>>>> 18096e65e14711a12028f98407498e5dfe43b02a
	$(CC) -shared -o $@ temp/*.o ${INCLUDE}

clean:
	rm -rf temp
	rm libfastsimilarity.so
