CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=mersenne.cpp  StatFunctions.cpp  TSHISAv1.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=stihiv

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
	
clean:
	rm $(OBJECTS) $(EXECUTABLE)
