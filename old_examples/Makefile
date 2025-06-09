A2D_INCLUDE=-I${HOME}/git/a2d/include
CFLAGS=-std=c++17 -g
# include ../Makefile.in

INCLUDE_DIRS = ../include
HEADERS := $(addsuffix /*.h, ${INCLUDE_DIRS})

INCLUDE_FLAGS = ${A2D_INCLUDE} -I../include
# LD_FLAGS = 

default: cart_pole

%.o: %.cpp ${HEADERS}
	${CXX} ${CFLAGS} ${INCLUDE_FLAGS} -c $< -o $*.o

cart_pole: ${HEADERS} cart_pole.o
	${CXX} ${CFLAGS} cart_pole.o ${LD_FLAGS} -o cart_pole

clean:
	rm -rf *.o cart_pole