ifeq (${REX_INSTALL},)
	REX_INSTALL := ${REX_ROOT}/rex_install
endif

default: main.cpp
	g++ main.cpp -I${REX_INSTALL}/include/rose -L${REX_INSTALL}/lib -lrose -lboost_system -o pfg.out

clean: 
	rm -rf *.o *.out rose*

