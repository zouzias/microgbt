
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
      13 text files.
      13 unique files.                              
       0 files ignored.

github.com/AlDanial/cloc v 1.84  T=0.01 s (869.5 files/s, 76110.5 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C/C++ Header                    10            199            274            587
C++                              2              8              3             47
CMake                            1              4              0             16
-------------------------------------------------------------------------------
SUM:                            13            211            277            650
-------------------------------------------------------------------------------
```
