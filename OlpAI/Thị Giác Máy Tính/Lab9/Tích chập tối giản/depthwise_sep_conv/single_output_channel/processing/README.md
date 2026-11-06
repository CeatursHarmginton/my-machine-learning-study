# Depthwise Separable Convolution Visualization (20x20x1 output)

This updated Processing sketch changes the final output to:

```text
20 x 20 x 1
```

## Important change

To keep the output spatial size at `20 x 20` while still using a `3 x 3` depthwise convolution with stride `1`, the sketch now uses:

```text
padding = 1
```

So the tensor flow becomes:

```text
Input tensor:                20 x 20 x 6
Spatial Filtering output:    20 x 20 x 6
Channel Combining output:    20 x 20 x 1
```

## Spatial Filtering

- 6 separate depthwise kernels
- each kernel is `3 x 3`
- one convolution per input channel
- SAME padding preserves width and height

## Channel Combining

- one pointwise combining kernel (`1 x 1 x 6`)
- combines the 6 intermediate channels into 1 final output channel
- the 6 combining weights are shown explicitly as voxels between the
  Spatial Filtering stack and the output

## Visual design

- uniform voxel sizes
- values represented by color only
- **all weights are shown explicitly as voxels**: the 6 depthwise kernels
  and the pointwise combining kernel (blue = positive, pink = negative)
- products physically travel through the weight voxels: a value moves into
  the weight, is multiplied there, and the product continues to its target
- separate sections for Spatial Filtering and Channel Combining
- planes rotated 90 degrees counterclockwise on the Y axis

## Run

1. Install **Processing 4**
2. Open `Depthwise_Separable_Convolution_20x20x1.pde`
3. Make sure **Java Mode** is selected
4. Run the sketch

## Controls

- `Space`: play/pause
- `Right Arrow`: complete the current spatial position
- `R`: reset
- `N`: generate new random data
- `V`: show/hide numeric overlay
- `[` / `]`: decrease/increase animation speed
- Mouse drag: orbit camera
- Mouse wheel: zoom
- `1`, `2`, `3`: preset views
