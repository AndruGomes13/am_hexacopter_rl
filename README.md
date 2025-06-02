<!-- README.md -->

# Project Setup

This README covers the steps to:

1. Create the Conda environment.
2. (Optional) Generate MuJoCo type‐hinting stubs—**not needed** if `typings/` already exists.
3. (Optional) Configure Pyright—**not needed** if `pyrightconfig.json` is already present.
4. (Optional) Configure Pylint—**not needed** if `.pylintrc` is already present.


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
```json
{
  "typeCheckingMode": "basic",
  "extraPaths": [
    "./typings"
  ]
}
```

## 4. Pylint Configuration (Not needed)
Add a .pylintrc at the project root to silence missing‐module errors for MuJoCo:
```ini
[TYPECHECK]
ignored-modules=mujoco,mujoco.mjx
```

