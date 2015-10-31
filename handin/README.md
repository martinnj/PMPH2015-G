# Group Project

# File structure
This archive contains the report as a PDF file called report.pdf,
as well as a folder called ```src```. This folder contains the needed files
to built the project.

The source folder includes 2 folders that are shared between the different
versions of our code: ```include``` and ```Data```.

All the different versions contain a makefile which supports the ```clean``` and
```run_*``` options the handout does. The compile command is specific for some
versions, but simply running ```make``` will compile and run the program for all
versions.

```00_HandoutImpl``` contains the version of the code that was given to us, with
one modification: The outer loop is parallelized using OpenMP.

```02_HandoutExpanded``` contains the version of the code where array expansion,
loop distribution as well as other transformations are applied, this version
also uses OpenMP to parallelize some of the loops.

```04_FlatArrayImpl``` contains the version where all vectors have been replaced
with linear arrays, version is the one that we translated into CUDA. It also
contains OpenMP loops to speed up any test executions.

```06_gpu_flat``` is our (unfinished) CUDA implementation. It supports the
common ```make``` targets as well as ```make gpu``` which will compile the code
without running it.