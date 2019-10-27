
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
      14 text files.
      14 unique files.                              
       0 files ignored.

github.com/AlDanial/cloc v 1.84  T=0.01 s (964.5 files/s, 81020.1 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C/C++ Header                    11            212            270            616
C++                              2              8              3             47
CMake                            1              4              0             16
-------------------------------------------------------------------------------
SUM:                            14            224            273            679
-------------------------------------------------------------------------------
```
