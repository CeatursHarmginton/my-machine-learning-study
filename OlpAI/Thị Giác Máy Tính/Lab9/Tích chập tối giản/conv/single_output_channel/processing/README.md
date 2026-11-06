# 3D Convolution Visualization — Uniform Voxel Version

This version keeps the previous **90-degree counterclockwise rotation around the Y axis** and changes the visual encoding so that:

- Every input voxel has the same size.
- Every temporary kernel/product voxel has the same size.
- Every output voxel has the same size and depth.
- Numerical values are represented by **color only**, not geometry.

## Weights are shown explicitly

The `3 x 3 x 6` kernel weights sit in the middle as colored voxels. During
each step the selected input values travel INTO the weight voxels, are
multiplied there, and the resulting products continue on to the single
output cell where they are summed.

## Color encoding

### Input and output values

The normalized value scale runs from:

```text
low value      middle value       high value
blue       ->  cyan           ->  yellow
```

### Kernel weights

- Cyan tones: positive weights
- Pink tones: negative weights

### Moving kernel products

- Cyan: positive product
- Pink: negative product
- Stronger opacity: larger product magnitude

## Tensor configuration

```text
Input:    20 x 20 x 6
Kernel:    3 x 3 x 6
Stride:    1
Padding:   0
Output:   18 x 18 x 1
```

## Run on Windows

1. Install Processing 4.
2. Extract the ZIP file.
3. Open `CNN_Convolution_3D_20x20x6_uniform_voxels.pde`.
4. Select Java Mode.
5. Click Run or press `Ctrl+R`.

## Controls

- `Space`: play or pause
- `Right Arrow`: advance to the next output cell
- `R`: reset
- `N`: generate new values
- `V`: show or hide channel contributions
- `[` and `]`: change speed
- Mouse drag: rotate the scene
- Mouse wheel: zoom
- `1`, `2`, `3`: preset views
