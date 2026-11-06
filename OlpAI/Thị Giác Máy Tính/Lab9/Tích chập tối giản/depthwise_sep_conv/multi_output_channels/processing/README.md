# Depthwise Separable Convolution Visualization — Multiple Output Channels (20x20x6 output)

This is the multi-output-channel version of the single-output depthwise
separable convolution sketch. The final output is now:

```text
20 x 20 x 6
```

## Key idea

A depthwise separable convolution has two stages:

1. **Spatial Filtering (depthwise)** — one `3 x 3` kernel per input channel.
   This stage is **unchanged** from the single-output version.
2. **Channel Combining (pointwise)** — this is where output channels are
   created. Each output channel needs its own pointwise `1 x 1 x 6`
   combining kernel. To get **6 output channels** we therefore use **6
   separate channel-combining kernels**.

To keep the spatial size at `20 x 20` with a `3 x 3` depthwise convolution
and stride `1`, the sketch uses `padding = 1` (SAME padding).

## Tensor flow

```text
Input tensor:                20 x 20 x 6
Spatial Filtering output:    20 x 20 x 6   (6 depthwise kernels)
Channel Combining output:    20 x 20 x 6   (6 pointwise kernels)
```

## Spatial Filtering

- 6 separate depthwise kernels
- each kernel is `3 x 3`
- one convolution per input channel
- SAME padding preserves width and height

## Channel Combining

- **6 pointwise combining kernels**, each `1 x 1 x 6`
- kernel `oc` combines the 6 intermediate channels into output channel `oc`
- produces `20 x 20 x 6`

## What is different from the single-output version

- The single pointwise combination became a **stack of 6 combining
  kernels** (shown between the Spatial Filtering stack and the output).
- The final output tensor now has **6 stacked planes** instead of one.
- For every position, after the 6 depthwise voxels are computed, the
  animation loops through all 6 combining kernels one at a time. The active
  combining kernel is highlighted in orange, and its weighted contributions
  fly into the matching output channel.

## Visual design

- uniform voxel sizes
- values represented by color only
- explicit depthwise kernels and explicit pointwise kernels are shown
- separate sections for Spatial Filtering and Channel Combining
- planes rotated 90 degrees counterclockwise on the Y axis

## Run

1. Install **Processing 4**
2. Open `Depthwise_Separable_Convolution_20x20x6.pde`
3. Make sure **Java Mode** is selected
4. Run the sketch

## Controls

- `Space`: play/pause
- `Right Arrow`: complete the current spatial position
- `R`: reset
- `N`: generate new random data
- `V`: show/hide numeric overlay for the active output channel
- `[` / `]`: decrease/increase animation speed
- Mouse drag: orbit camera
- Mouse wheel: zoom
- `1`, `2`, `3`: preset views
