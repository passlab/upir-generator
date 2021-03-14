ifeq (${REX_INSTALL},)
	REX_INSTALL := ${REX_ROOT}/rex_install
endif

default: main.cpp ./flow_graph/flow_graph.cpp
	g++ main.cpp ./flow_graph/flow_graph.cpp -I ./flow_graph -I${REX_INSTALL}/include/rose -L${REX_INSTALL}/lib -lrose -lboost_system -o pfg.out

clean: 
	rm -rf *.o *.out rose*

