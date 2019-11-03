
## Python wheels packaging

```bash
python setup.py bdist_wheel
ls dist/
microgbtpy-0.0.1-XXX
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

github.com/AlDanial/cloc v 1.84  T=0.02 s (797.5 files/s, 77178.1 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C/C++ Header                    10            233            246            701
C++                              2              8              3             47
CMake                            1              4              0             16
-------------------------------------------------------------------------------
SUM:                            13            245            249            764
-------------------------------------------------------------------------------

```
