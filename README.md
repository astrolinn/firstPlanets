# firstPlanets

**Code developed for the paper "Planets and planetesimals at cosmic dawn: Vortices as planetary nurseries" (Eriksson et al. 2025)**

Dust evolution is performed with DustPy (Stammler & Birnstiel 2022).
The effect of vortices is modeled via post-processing. 
Streaming instability criteria are used to determine where and when embryos/planetesimals form. 
The formed embryos continue to grow via the accretion of pebbles, with varying assumptions for the residence time of the embryos within the vortex. 

---

## Requirements

You need to install the DustPy code:

- **DustPy**: https://github.com/stammler/dustpy

### Additional Python dependencies:

- `matplotlib`

(Other dependencies will typically be installed by or at the same time as the dust evolution code.)

---

## How to Run the Code

1. **Create a simulation directory**, and copy `inputFile.py` into it.
2. **Edit `inputFile.py`** to set your simulation parameters. The default parameters are the same that are used in Fig.1 of Eriksson et al. (2025). 
3. From the main `firstPlanets` directory, run the following (dust evolution needs to finish before post-processing can be run):

```bash
python dustev.py --path pathToDir
python -u main.py --path pathToDir
```
---

If you use this code, cite Eriksson et al. (2025).
