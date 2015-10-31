CXX        = nvcc
CXXFLAGS   = -O3 -DWITH_FLOATS=1 -arch=sm_20

INCLUDES  += -I ../include
GPU_OPTS   = -D lgWARP=5

SOURCES_CPP =ProjectMain.cu ProjHelperFun.cu ProjCoreOrig.cu
#HELPERS     =ProjhHelperFun.cu.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/TridagPar.h ../include/TridagKernel.cu.h
#OBJECTS     =ProjectMain.cu.o ProjHelperFun.cu.o  ProjCoreOrig.cu.o
EXECUTABLE  =runproject


default: clean gpu run_small

#.cpp.o: $(SOURCES_CPP) $(HELPERS)
#	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

#.cu.o: $(SOURCES_CPP) $(HELPERS)
#	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

gpu: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) $(SOURCES_CPP)


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE)