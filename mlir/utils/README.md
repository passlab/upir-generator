
This module traverses the OpenMP statements and collects the usage information of parallel data.

Given input:

```c
void axpy (float* x, float* y, float a, int n) {
    int i;
#pragma omp parallel for num_threads(6)
    for (i = 0; i < n; i++) {
        y[i] = y[i] + a * x[i];
    }
}
```

The output would be:

```bash
...
Variable symbol: "x"
	Sharing property: shared
	Sharing visibility: implicit
	Mapping property: 
	Mapping visibility: 
	Data access: read-only
Variable symbol: "y"
	Sharing property: shared
	Sharing visibility: implicit
	Mapping property: 
	Mapping visibility: 
	Data access: read-write
Variable symbol: "a"
	Sharing property: shared
	Sharing visibility: implicit
	Mapping property: 
	Mapping visibility: 
	Data access: read-only
Variable symbol: "n"
	Sharing property: shared
	Sharing visibility: implicit
	Mapping property: 
	Mapping visibility: 
	Data access: read-only
Variable symbol: "i"
	Sharing property: private
	Sharing visibility: implicit
	Mapping property: 
	Mapping visibility: 
	Data access: read-write
...
```
