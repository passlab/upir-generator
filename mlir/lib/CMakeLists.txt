add_mlir_dialect_library(MLIRUpirDialect
        Dialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/upir

        DEPENDS
        UpirOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
