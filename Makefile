ifeq (${REX_INSTALL},)
	REX_INSTALL := ${REX_ROOT}/rex_install
endif

default: main.cpp ./flow_graph/flow_graph.cpp ./visualization/visualization.cpp
	g++ main.cpp ./flow_graph/flow_graph.cpp ./visualization/visualization.cpp -I./flow_graph -I./visualization -I${REX_INSTALL}/include/rose -L${REX_INSTALL}/lib -lrose -lboost_system -o pfg.out

clean: 
	rm -rf *.o *.out rose*

