
## Python wheels packaging

```bash
python setup.py bdist_wheel
ls dist/
microgbtpy-0.0.1-cp36-cp36m-macosx_10_13_x86_64.whl
```

## To count lines of code 

Type, (`brew install cloc` to install cloc)

```bash
cloc src
```

returns

```text
     12 text files.
      12 unique files.                              
       0 files ignored.

github.com/AlDanial/cloc v 1.82  T=0.01 s (809.9 files/s, 73226.7 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C/C++ Header                     9            191            252            561
C++                              2             10              3             48
CMake                            1              4              0             16
-------------------------------------------------------------------------------
SUM:                            12            205            255            625
-------------------------------------------------------------------------------

```
