CC=gcc
#INCLUDE=
CFLAGS=-Wall -fPIC -O3

all: fastsimilarity.so

temp:  
	mkdir temp

temp/fastsimilarity.o: temp
	$(CC) -c similarity.c -o $@ ${CFLAGS} ${INCLUDE}

fastsimilarity.so: temp/fastsimilarity.o 
	$(CC) -shared -o $@ temp/*.o ${INCLUDE}

clean:
	rm -rf temp
	rm fastsimilarity.so
