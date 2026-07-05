# Glass-Box MLP Simulations

This folder contains interactive Plotly simulations for teaching how small multilayer perceptrons (MLPs) learn from data.

The simulations are designed for students who are learning the forward pass, activation functions, loss calculation, backpropagation, and gradient-based parameter updates.

## Files

- `mlp121.py`: creates an animation for a `1-2-1` MLP.
- `mlp1221.py`: creates an animation for a `1-2-2-1` MLP.
- `glass_box_mlp.html`: generated HTML animation for the `1-2-1` network.
- `glass_box_mlp_1221.html`: generated HTML animation for the `1-2-2-1` network.
- `document/mlp_student_assignments.md`: theory notes and student assignments for both networks.

## Requirements

Install Python packages:

```powershell
pip install numpy plotly
```

## Generate The 1-2-1 Animation

```powershell
python mlp121.py --output glass_box_mlp.html --epochs 5 --lr 0.08
```

To generate the file without automatically opening it:

```powershell
python mlp121.py --output glass_box_mlp.html --epochs 5 --lr 0.08 --no-open
```

## Generate The 1-2-2-1 Animation

```powershell
python mlp1221.py --output glass_box_mlp_1221.html --epochs 5 --lr 0.08
```

To generate the file without automatically opening it:

```powershell
python mlp1221.py --output glass_box_mlp_1221.html --epochs 5 --lr 0.08 --no-open
```

## Suggested Teaching Flow

1. Show students the `1-2-1` simulation.
2. Ask them to complete the `1-2-1` theory and calculation assignments.
3. Show students the `1-2-2-1` simulation.
4. Ask them to complete the deeper `1-2-2-1` assignments.

The assignment document is in:

```text
document/mlp_student_assignments.md
```

## Notes

- The animations use a small synthetic dataset generated inside each script.
- The models use squared error loss.
- Hidden layers use ReLU activation.
- The output neuron is linear, so the network can predict continuous numeric values.
