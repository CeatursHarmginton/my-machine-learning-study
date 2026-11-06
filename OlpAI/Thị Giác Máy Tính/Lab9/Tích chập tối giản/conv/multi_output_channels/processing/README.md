# 3D Convolution Visualization — Multiple Output Channels

This is the multi-output-channel version of the single-output normal
convolution sketch. A standard convolution layer produces **one output
channel per kernel**, so to obtain **6 output channels** we use **6
independent kernels**, each shaped `3 x 3 x 6`.

Every kernel slides over the whole `20 x 20 x 6` input and writes into its
own `18 x 18` output plane, giving a final output of `18 x 18 x 6`.

## What is different from the single-output version

- The single kernel became a **stack of 6 kernels** (shown in the middle).
- The output tensor now has **6 stacked planes** instead of one.
- For every output position the animation loops through all 6 kernels
  one at a time. The currently active kernel is highlighted in orange, and
  its 54 products fly into the matching output channel.

## Color encoding

### Input and output values

```text
low value      middle value       high value
blue       ->  cyan           ->  yellow
```

### Kernels

- Cyan tones: positive weights
- Pink tones: negative weights

### Moving kernel products

- Cyan: positive product
- Pink: negative product
- Stronger opacity: larger product magnitude

## Tensor configuration

```text
Input:    20 x 20 x 6
Kernels:  6 x (3 x 3 x 6)
Stride:   1
Padding:  0
Output:   18 x 18 x 6
```

Each output channel size: `O = (20 - 3) / 1 + 1 = 18`.

## Run on Windows

1. Install Processing 4.
2. Open `CNN_Convolution_3D_20x20x6_multi_output.pde`.
3. Select Java Mode.
4. Click Run or press `Ctrl+R`.

## Controls

- `Space`: play or pause
- `Right Arrow`: advance to the next kernel / output cell
- `R`: reset
- `N`: generate new values
- `V`: show or hide channel contributions for the active output channel
- `[` and `]`: change speed
- Mouse drag: rotate the scene
- Mouse wheel: zoom
- `1`, `2`, `3`: preset views
