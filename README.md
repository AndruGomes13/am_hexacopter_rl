<!-- README.md -->

# Project Setup

This README covers the steps to:

1. Create the Conda environment.
2. Generate MuJoCo type‐hinting stubs (assuming `typings/` is already in the repo).
3. Configure Pyright and Pylint to use those stubs.

---

## 1. Create the Conda Environment

If you haven’t already created the environment, run:

```bash
conda env create -f environment.yml
```

Then activate it:
```bash
conda activate am-hexacopter-rl
```

## 2. Generate MuJoCo Type Hints (Not needed)

```bash
pybind11-stubgen mujoco -o ~/typings/ -o ./typings
pybind11-stubgen mujoco.mjx -o ./typings
```

## 3. Pyright Configuration (Not needed)
Create a pyrightconfig.json in the project root (if it’s not already there):
```bash
{
  "typeCheckingMode": "basic",
  "extraPaths": [
    "./typings"
  ]
}
```

## 4. Pylint Configuration (Not needed)
Add a .pylintrc at the project root to silence missing‐module errors for MuJoCo:
```bash
[TYPECHECK]
ignored-modules=mujoco,mujoco.mjx
```


