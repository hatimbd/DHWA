# Spiking Neural Networks Dynamic HandWriting Authentication


## Installation

_conda create -f environment.yml_


## Paths

**Make sure that you have th right paths in the beginning of _authentication.py_**


## Use

```bash
_conda activate snndhwa_
```
```python
_python authentication.py_
```
### Files Guide
- dhwa.py : code fourni par le prof (HIDDEN= 1000 et BATCH = 8)  
- dhwa_cpu.py : "OPTIMISATION CRITIQUE CPU + OPTIMISATION ALGORITHMIQUE" sur tout le DATASET (avec HIDDEN= 128 et BATCH = 32)    
- dhwa_fourletters.py : le meme que dhwa_cpu.py sauf que c'est sur que 4 lettres. (avec HIDDEN= 128 et BATCH = 8)  
- dhwa_fourletters_.py : le meme que dhwa.py sauf que c'est sur que 4 lettres (avec HIDDEN= 128 et BATCH = 32)
- dhwa_numbers.py : copie de dhwa_fourletters.py en utilisant des chiffres au lieu de lettres