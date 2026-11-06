import processing.event.MouseEvent;

/*
  Depthwise separable convolution visualization - MULTIPLE OUTPUT CHANNELS

  Final output: 20 x 20 x 6

  To keep the spatial size at 20 x 20 with a 3 x 3 filter and stride 1,
  this sketch uses SAME padding (padding = 1).

  Stage 1: Spatial Filtering (depthwise)
    Input:   20 x 20 x 6
    Kernels: 6 separate 3 x 3 kernels, one per input channel
    Stride:  1
    Padding: 1
    Output:  20 x 20 x 6   (one intermediate plane per input channel)

  Stage 2: Channel Combining (pointwise)
    Input:   20 x 20 x 6
    Kernels: 6 separate pointwise (1 x 1 x 6) combining kernels,
             one per output channel
    Output:  20 x 20 x 6

  This is the multi-output-channel version of the single-output sketch.
  The depthwise stage is unchanged; the change is that Channel Combining
  now uses 6 pointwise kernels instead of one, so it produces 6 output
  channels instead of 1.

  This version keeps:
  - explicit multiple depthwise kernels
  - explicit multiple channel-combining (pointwise) kernels
  - separate Spatial Filtering and Channel Combining sections
  - uniform voxel sizes
  - values represented by color
  - tensor planes rotated 90 degrees counterclockwise on the Y axis
*/

final int INPUT_SIZE = 20;
final int IN_CHANNELS = 6;
final int KERNEL_SIZE = 3;
final int STRIDE = 1;
final int PADDING = 1;
final int MID_SIZE = 20;      // SAME padding keeps spatial size
final int OUT_CHANNELS = 6;

final float ROTATE_Y_CCW = HALF_PI;

final float INPUT_STEP = 6.0;
final float KERNEL_STEP = 10.5;
final float MID_STEP = 6.2;
final float OUT_STEP = 6.2;

final float IN_LAYER_GAP = 16.0;
final float KERNEL_LAYER_GAP = 16.0;
final float MID_LAYER_GAP = 16.0;
final float OUT_LAYER_GAP = 15.0;

// pointwise combining kernels: 6 kernels, each 1 x 1 x 6
final float PW_CH_GAP = 8.5;      // spacing between the 6 weights of one kernel
final float PW_SLOT = 26.0;       // spacing between the 6 combining kernels
final float PW_VOXEL_SIZE = 6.4;

final float VOXEL_SIZE = 5.2;
final float KERNEL_VOXEL_SIZE = 6.0;
final float MOVING_VOXEL_SIZE = 5.6;

float[][][] inputTensor = new float[IN_CHANNELS][INPUT_SIZE][INPUT_SIZE];
float[][][] depthwiseKernel = new float[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
float[][][] spatialOut = new float[IN_CHANNELS][MID_SIZE][MID_SIZE];
float[][] pointwiseWeights = new float[OUT_CHANNELS][IN_CHANNELS];
float[][][] finalOut = new float[OUT_CHANNELS][MID_SIZE][MID_SIZE];

float[][][] spatialNorm = new float[IN_CHANNELS][MID_SIZE][MID_SIZE];
float[][][] finalNorm = new float[OUT_CHANNELS][MID_SIZE][MID_SIZE];
float[][][] kernelNorm = new float[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
float[][] pointwiseNorm = new float[OUT_CHANNELS][IN_CHANNELS];

boolean[][][] spatialRevealed = new boolean[IN_CHANNELS][MID_SIZE][MID_SIZE];
boolean[][][] finalRevealed = new boolean[OUT_CHANNELS][MID_SIZE][MID_SIZE];

PVector inputOrigin = new PVector(-560, 0, -25);
PVector kernelOrigin = new PVector(-280, -4, -5);
PVector spatialOrigin = new PVector(30, 0, 0);
PVector pointwiseOrigin = new PVector(330, -4, 0);
PVector finalOrigin = new PVector(560, 0, 10);

int outRow = 0;
int outCol = 0;
int depthChannel = 0;
int outputChannel = 0;

// phases:
// 0 spatial inspect
// 1 spatial transfer
// 2 channel-combine inspect
// 3 channel-combine transfer
// 4 complete
int phase = 0;
float phaseTime = 0;
boolean playing = true;
boolean showValues = false;
float animationSpeed = 1.4;

float yaw = -0.47;
float pitch = 0.40;
float zoomAmount = 0.80;
float previousMouseX;
float previousMouseY;
int lastFrameMillis;

color bgColor;
color wireColor;
color mutedColor;
color activeColor;
color positiveColor;
color negativeColor;

void settings() {
  size(1680, 900, P3D);
  smooth(8);
}

void setup() {
  surface.setTitle("Depthwise Separable Convolution 20x20x6");
  frameRate(60);

  bgColor = color(4, 6, 10);
  wireColor = color(228, 234, 240);
  mutedColor = color(76, 88, 100);
  activeColor = color(255, 190, 60);
  positiveColor = color(90, 228, 255);
  negativeColor = color(255, 110, 150);

  textFont(createFont("Arial", 14));

  randomizeData();
  resetAnimation();
  lastFrameMillis = millis();
}

void draw() {
  int now = millis();
  float dt = min((now - lastFrameMillis) / 1000.0, 0.05);
  lastFrameMillis = now;

  updateAnimation(dt);

  background(bgColor);
  perspective(PI / 3.1, float(width) / float(height), 5, 6000);

  ambientLight(74, 74, 80);
  directionalLight(215, 220, 230, -0.25, 0.45, -1.0);

  pushMatrix();
  translate(width * 0.50, height * 0.58, 0);
  scale(zoomAmount);
  rotateX(pitch);
  rotateY(yaw);

  drawFloorGrid();
  drawTensorFrames();
  drawInputTensor();
  drawDepthwiseKernelStack();
  drawSpatialTensor();
  drawPointwiseKernelStack();
  drawFinalTensor();
  drawTransferGuides();
  drawMovingSpatialOperation();
  drawMovingCombineOperation();

  popMatrix();

  drawHUD();
}

void randomizeData() {
  noiseSeed((int)random(100000));

  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    float off = 13.0 * ch;
    for (int r = 0; r < INPUT_SIZE; r++) {
      for (int c = 0; c < INPUT_SIZE; c++) {
        float n1 = noise(c * 0.15, r * 0.15, off);
        float n2 = noise(c * 0.35 + 18, r * 0.35 + 9, off + 7.3);
        inputTensor[ch][r][c] = constrain(0.68 * n1 + 0.32 * n2, 0, 1);
      }
    }
  }

  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
      for (int kc = 0; kc < KERNEL_SIZE; kc++) {
        float radial = 1.0 - dist(kc, kr, 1, 1) / 1.8;
        float sign = (ch % 2 == 0) ? 1.0 : -1.0;
        depthwiseKernel[ch][kr][kc] = sign * 0.30 * radial + random(-0.55, 0.55);
      }
    }
  }

  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    for (int ch = 0; ch < IN_CHANNELS; ch++) {
      float base = map(ch, 0, IN_CHANNELS - 1, -0.45, 0.65);
      float ocSign = (oc % 2 == 0) ? 1.0 : -1.0;
      pointwiseWeights[oc][ch] = ocSign * base + random(-0.30, 0.30);
    }
  }

  computeSpatialStage();
  computeFinalStage();
  normalizeSpatial();
  normalizeFinal();
  normalizeKernels();
  normalizePointwise();
}

float paddedInput(int ch, int r, int c) {
  if (r < 0 || r >= INPUT_SIZE || c < 0 || c >= INPUT_SIZE) {
    return 0;
  }
  return inputTensor[ch][r][c];
}

void computeSpatialStage() {
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        float sum = 0;
        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
          for (int kc = 0; kc < KERNEL_SIZE; kc++) {
            int inputR = r + kr - PADDING;
            int inputC = c + kc - PADDING;
            sum += paddedInput(ch, inputR, inputC) * depthwiseKernel[ch][kr][kc];
          }
        }
        spatialOut[ch][r][c] = sum;
      }
    }
  }
}

void computeFinalStage() {
  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        float sum = 0;
        for (int ch = 0; ch < IN_CHANNELS; ch++) {
          sum += spatialOut[ch][r][c] * pointwiseWeights[oc][ch];
        }
        finalOut[oc][r][c] = sum;
      }
    }
  }
}

void normalizeSpatial() {
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    float mn = Float.MAX_VALUE;
    float mx = -Float.MAX_VALUE;
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        mn = min(mn, spatialOut[ch][r][c]);
        mx = max(mx, spatialOut[ch][r][c]);
      }
    }
    float range = max(mx - mn, 0.0001);
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        spatialNorm[ch][r][c] = (spatialOut[ch][r][c] - mn) / range;
      }
    }
  }
}

void normalizeFinal() {
  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    float mn = Float.MAX_VALUE;
    float mx = -Float.MAX_VALUE;
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        mn = min(mn, finalOut[oc][r][c]);
        mx = max(mx, finalOut[oc][r][c]);
      }
    }
    float range = max(mx - mn, 0.0001);
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        finalNorm[oc][r][c] = (finalOut[oc][r][c] - mn) / range;
      }
    }
  }
}

void normalizeKernels() {
  float absMax = 0.0001;
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
      for (int kc = 0; kc < KERNEL_SIZE; kc++) {
        absMax = max(absMax, abs(depthwiseKernel[ch][kr][kc]));
      }
    }
  }
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
      for (int kc = 0; kc < KERNEL_SIZE; kc++) {
        kernelNorm[ch][kr][kc] = map(depthwiseKernel[ch][kr][kc], -absMax, absMax, 0, 1);
      }
    }
  }
}

void normalizePointwise() {
  float absMax = 0.0001;
  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    for (int ch = 0; ch < IN_CHANNELS; ch++) {
      absMax = max(absMax, abs(pointwiseWeights[oc][ch]));
    }
  }
  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    for (int ch = 0; ch < IN_CHANNELS; ch++) {
      pointwiseNorm[oc][ch] = map(pointwiseWeights[oc][ch], -absMax, absMax, 0, 1);
    }
  }
}

void resetAnimation() {
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        spatialRevealed[ch][r][c] = false;
      }
    }
  }

  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        finalRevealed[oc][r][c] = false;
      }
    }
  }

  outRow = 0;
  outCol = 0;
  depthChannel = 0;
  outputChannel = 0;
  phase = 0;
  phaseTime = 0;
  playing = true;
}

void updateAnimation(float dt) {
  if (!playing || phase == 4) return;

  phaseTime += dt * animationSpeed;

  if (phase == 0 && phaseTime >= 0.40) {
    phase = 1;
    phaseTime = 0;
  } else if (phase == 1 && phaseTime >= 0.95) {
    spatialRevealed[depthChannel][outRow][outCol] = true;
    phaseTime = 0;
    if (depthChannel < IN_CHANNELS - 1) {
      depthChannel++;
      phase = 0;
    } else {
      depthChannel = 0;
      outputChannel = 0;
      phase = 2;
    }
  } else if (phase == 2 && phaseTime >= 0.34) {
    phase = 3;
    phaseTime = 0;
  } else if (phase == 3 && phaseTime >= 0.85) {
    finalRevealed[outputChannel][outRow][outCol] = true;
    phaseTime = 0;
    if (outputChannel < OUT_CHANNELS - 1) {
      outputChannel++;
      phase = 2;
    } else {
      outputChannel = 0;
      advanceSpatialCursor();
    }
  }
}

void advanceSpatialCursor() {
  outCol++;
  if (outCol >= MID_SIZE) {
    outCol = 0;
    outRow++;
  }

  if (outRow >= MID_SIZE) {
    outRow = MID_SIZE - 1;
    outCol = MID_SIZE - 1;
    phase = 4;
    playing = false;
  } else {
    phase = 0;
  }
}

void completeCurrentPosition() {
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    spatialRevealed[ch][outRow][outCol] = true;
  }
  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    finalRevealed[oc][outRow][outCol] = true;
  }
  depthChannel = 0;
  outputChannel = 0;
  phaseTime = 0;
  advanceSpatialCursor();
}

void drawInputTensor() {
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    for (int r = 0; r < INPUT_SIZE; r++) {
      for (int c = 0; c < INPUT_SIZE; c++) {
        boolean highlighted = (phase == 0 || phase == 1)
          && ch == depthChannel
          && c >= max(0, outCol - PADDING) && c <= min(INPUT_SIZE - 1, outCol - PADDING + 2)
          && r >= max(0, outRow - PADDING) && r <= min(INPUT_SIZE - 1, outRow - PADDING + 2);

        PVector p = inputPosition(ch, r, c);
        color vc = valueColor(inputTensor[ch][r][c]);

        pushMatrix();
        translate(p.x, p.y, p.z);
        stroke(highlighted ? activeColor : wireColor,
               highlighted ? 255 : 36);
        strokeWeight(highlighted ? 1.3 : 0.55);
        fill(vc, highlighted ? 195 : 78);
        box(VOXEL_SIZE, VOXEL_SIZE, 3.0);
        popMatrix();
      }
    }
  }
}

void drawDepthwiseKernelStack() {
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    boolean activeKernel = (phase == 0 || phase == 1) && ch == depthChannel;

    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
      for (int kc = 0; kc < KERNEL_SIZE; kc++) {
        PVector p = depthwiseKernelPosition(ch, kr, kc);
        color vc = kernelColor(depthwiseKernel[ch][kr][kc], kernelNorm[ch][kr][kc]);

        pushMatrix();
        translate(p.x, p.y, p.z);
        stroke(activeKernel ? activeColor : wireColor, activeKernel ? 210 : 70);
        strokeWeight(activeKernel ? 1.2 : 0.8);
        fill(vc, activeKernel ? 225 : 165);
        box(KERNEL_VOXEL_SIZE, KERNEL_VOXEL_SIZE, 4.0);
        popMatrix();
      }
    }
  }
}

void drawPointwiseKernelStack() {
  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    boolean activeKernel = (phase == 2 || phase == 3) && oc == outputChannel;

    for (int ch = 0; ch < IN_CHANNELS; ch++) {
      PVector p = pointwiseKernelPosition(oc, ch);
      color vc = kernelColor(pointwiseWeights[oc][ch], pointwiseNorm[oc][ch]);

      pushMatrix();
      translate(p.x, p.y, p.z);
      stroke(activeKernel ? activeColor : wireColor, activeKernel ? 215 : 65);
      strokeWeight(activeKernel ? 1.2 : 0.75);
      fill(vc, activeKernel ? 230 : 150);
      box(PW_VOXEL_SIZE, PW_VOXEL_SIZE, 4.2);
      popMatrix();
    }
  }
}

void drawSpatialTensor() {
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        PVector p = spatialPosition(ch, r, c);
        boolean currentSpatialTarget = (phase == 0 || phase == 1) && ch == depthChannel && r == outRow && c == outCol;
        boolean currentCombineSource = (phase == 2 || phase == 3) && r == outRow && c == outCol;

        pushMatrix();
        translate(p.x, p.y, p.z);

        if (spatialRevealed[ch][r][c]) {
          color vc = valueColor(spatialNorm[ch][r][c]);
          stroke(wireColor, currentCombineSource ? 200 : 120);
          strokeWeight(currentCombineSource ? 1.15 : 0.7);
          fill(vc, currentCombineSource ? 210 : 175);
          box(VOXEL_SIZE, VOXEL_SIZE, 3.2);
        } else if (currentSpatialTarget) {
          stroke(activeColor, 255);
          strokeWeight(1.35);
          fill(activeColor, 32);
          box(VOXEL_SIZE, VOXEL_SIZE, 3.2);
        } else {
          stroke(mutedColor, 80);
          strokeWeight(0.45);
          noFill();
          box(VOXEL_SIZE, VOXEL_SIZE, 2.6);
        }
        popMatrix();
      }
    }
  }
}

void drawFinalTensor() {
  for (int oc = 0; oc < OUT_CHANNELS; oc++) {
    for (int r = 0; r < MID_SIZE; r++) {
      for (int c = 0; c < MID_SIZE; c++) {
        PVector p = finalPosition(oc, r, c);
        boolean currentTarget = (phase == 2 || phase == 3) && oc == outputChannel && r == outRow && c == outCol;

        pushMatrix();
        translate(p.x, p.y, p.z);
        if (finalRevealed[oc][r][c]) {
          color vc = valueColor(finalNorm[oc][r][c]);
          stroke(wireColor, 130);
          strokeWeight(0.7);
          fill(vc, 190);
          box(VOXEL_SIZE, VOXEL_SIZE, 3.6);
        } else if (currentTarget) {
          stroke(activeColor, 255);
          strokeWeight(1.4);
          fill(activeColor, phase == 3 ? 105 : 28);
          float pulse = 1.0 + 0.12 * sin(frameCount * 0.13);
          box(VOXEL_SIZE * pulse, VOXEL_SIZE * pulse, 3.6);
        } else {
          stroke(mutedColor, 85);
          strokeWeight(0.45);
          noFill();
          box(VOXEL_SIZE, VOXEL_SIZE, 2.6);
        }
        popMatrix();
      }
    }
  }
}

void drawTransferGuides() {
  PVector inCenter = patchCenter(depthChannel);
  PVector kernelCenter = depthwiseKernelCenter(depthChannel);
  PVector spatialTarget = spatialPosition(depthChannel, outRow, outCol);
  stroke(activeColor, (phase == 0 || phase == 1) ? 120 : 45);
  strokeWeight(1.0);
  line(inCenter.x, inCenter.y, inCenter.z, kernelCenter.x, kernelCenter.y, kernelCenter.z);
  line(kernelCenter.x, kernelCenter.y, kernelCenter.z, spatialTarget.x, spatialTarget.y, spatialTarget.z);

  PVector stackCenter = spatialStackCenter();
  PVector pwCenter = pointwiseKernelCenter(outputChannel);
  PVector finalTarget = finalPosition(outputChannel, outRow, outCol);
  stroke(activeColor, (phase == 2 || phase == 3) ? 120 : 45);
  strokeWeight(1.0);
  line(stackCenter.x, stackCenter.y, stackCenter.z, pwCenter.x, pwCenter.y, pwCenter.z);
  line(pwCenter.x, pwCenter.y, pwCenter.z,
       finalTarget.x, finalTarget.y, finalTarget.z);

  pushMatrix();
  translate(pwCenter.x, pwCenter.y, pwCenter.z);
  rotateY(ROTATE_Y_CCW);
  noFill();
  stroke(activeColor, (phase == 2 || phase == 3) ? 130 : 45);
  box(14, 14, (IN_CHANNELS - 1) * PW_CH_GAP + 12);
  popMatrix();
}

void drawMovingSpatialOperation() {
  if (phase != 1) return;

  float progress = constrain(phaseTime / 0.95, 0, 1);
  float split = 0.58;

  for (int kr = 0; kr < KERNEL_SIZE; kr++) {
    for (int kc = 0; kc < KERNEL_SIZE; kc++) {
      int inputR = outRow + kr - PADDING;
      int inputC = outCol + kc - PADDING;
      PVector source;
      float inputVal;

      if (inputR < 0 || inputR >= INPUT_SIZE || inputC < 0 || inputC >= INPUT_SIZE) {
        source = paddedSourcePosition(depthChannel, inputR, inputC);
        inputVal = 0;
      } else {
        source = inputPosition(depthChannel, inputR, inputC);
        inputVal = inputTensor[depthChannel][inputR][inputC];
      }

      PVector mid = depthwiseKernelPosition(depthChannel, kr, kc);
      PVector target = spatialPosition(depthChannel, outRow, outCol);

      PVector pos;
      float tLocal;
      if (progress < split) {
        tLocal = easeInOut(progress / split);
        pos = interpolate(source, mid, tLocal);
      } else {
        tLocal = easeInOut((progress - split) / (1.0 - split));
        pos = interpolate(mid, target, tLocal);
      }

      float product = inputVal * depthwiseKernel[depthChannel][kr][kc];
      color pc = product >= 0 ? positiveColor : negativeColor;
      float alpha = 70 + 140 * constrain(abs(product) * 3.0, 0, 1);

      pushMatrix();
      translate(pos.x, pos.y, pos.z);
      stroke(pc, 215);
      strokeWeight(1.0);
      fill(pc, alpha);
      box(MOVING_VOXEL_SIZE, MOVING_VOXEL_SIZE, 4.0);
      popMatrix();
    }
  }
}

void drawMovingCombineOperation() {
  if (phase != 3) return;

  float progress = constrain(phaseTime / 0.85, 0, 1);
  float split = 0.55;

  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    PVector source = spatialPosition(ch, outRow, outCol);
    PVector mid = pointwiseKernelPosition(outputChannel, ch);
    PVector target = finalPosition(outputChannel, outRow, outCol);

    PVector pos;
    float tLocal;
    if (progress < split) {
      tLocal = easeInOut(progress / split);
      pos = interpolate(source, mid, tLocal);
    } else {
      tLocal = easeInOut((progress - split) / (1.0 - split));
      pos = interpolate(mid, target, tLocal);
    }

    float contribution = spatialOut[ch][outRow][outCol] * pointwiseWeights[outputChannel][ch];
    color cc = contribution >= 0 ? positiveColor : negativeColor;
    float alpha = 70 + 140 * constrain(abs(contribution) * 0.85, 0, 1);

    pushMatrix();
    translate(pos.x, pos.y, pos.z);
    stroke(cc, 215);
    strokeWeight(1.0);
    fill(cc, alpha);
    box(MOVING_VOXEL_SIZE, MOVING_VOXEL_SIZE, 4.0);
    popMatrix();
  }
}

void drawTensorFrames() {
  pushMatrix();
  translate(inputOrigin.x, inputOrigin.y, inputOrigin.z);
  rotateY(ROTATE_Y_CCW);
  stroke(wireColor, 90);
  strokeWeight(0.9);
  noFill();
  box((INPUT_SIZE - 1) * INPUT_STEP + 12,
      (INPUT_SIZE - 1) * INPUT_STEP + 12,
      (IN_CHANNELS - 1) * IN_LAYER_GAP + 10);
  popMatrix();

  pushMatrix();
  translate(kernelOrigin.x, kernelOrigin.y, kernelOrigin.z);
  rotateY(ROTATE_Y_CCW);
  stroke(wireColor, 90);
  strokeWeight(0.9);
  noFill();
  box((KERNEL_SIZE - 1) * KERNEL_STEP + 16,
      (KERNEL_SIZE - 1) * KERNEL_STEP + 16,
      (IN_CHANNELS - 1) * KERNEL_LAYER_GAP + 12);
  popMatrix();

  pushMatrix();
  translate(spatialOrigin.x, spatialOrigin.y, spatialOrigin.z);
  rotateY(ROTATE_Y_CCW);
  stroke(wireColor, 90);
  strokeWeight(0.9);
  noFill();
  box((MID_SIZE - 1) * MID_STEP + 12,
      (MID_SIZE - 1) * MID_STEP + 12,
      (IN_CHANNELS - 1) * MID_LAYER_GAP + 10);
  popMatrix();

  pushMatrix();
  translate(finalOrigin.x, finalOrigin.y, finalOrigin.z);
  rotateY(ROTATE_Y_CCW);
  stroke(wireColor, 90);
  strokeWeight(0.9);
  noFill();
  box((MID_SIZE - 1) * OUT_STEP + 12,
      (MID_SIZE - 1) * OUT_STEP + 12,
      (OUT_CHANNELS - 1) * OUT_LAYER_GAP + 10);
  popMatrix();
}

void drawFloorGrid() {
  stroke(95, 110, 122, 28);
  strokeWeight(0.6);
  float y = 150;
  for (int i = -12; i <= 12; i++) {
    line(i * 70, y, -340, i * 70, y, 340);
    line(-720, y, i * 30, 720, y, i * 30);
  }
}

PVector rotateLocalYCCW(float x, float y, float z) {
  return new PVector(z, y, -x);
}

PVector addOrigin(PVector origin, PVector local) {
  return new PVector(origin.x + local.x, origin.y + local.y, origin.z + local.z);
}

PVector inputPosition(int ch, int row, int col) {
  float lx = (col - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float ly = (row - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float lz = (ch - (IN_CHANNELS - 1) / 2.0) * IN_LAYER_GAP;
  return addOrigin(inputOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector paddedSourcePosition(int ch, int row, int col) {
  float clampRow = constrain(row, -1, INPUT_SIZE);
  float clampCol = constrain(col, -1, INPUT_SIZE);
  float lx = (clampCol - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float ly = (clampRow - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float lz = (ch - (IN_CHANNELS - 1) / 2.0) * IN_LAYER_GAP;
  return addOrigin(inputOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector depthwiseKernelPosition(int ch, int kr, int kc) {
  float lx = (kc - (KERNEL_SIZE - 1) / 2.0) * KERNEL_STEP;
  float ly = (kr - (KERNEL_SIZE - 1) / 2.0) * KERNEL_STEP;
  float lz = (ch - (IN_CHANNELS - 1) / 2.0) * KERNEL_LAYER_GAP;
  return addOrigin(kernelOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector pointwiseKernelPosition(int oc, int ch) {
  float lx = 0;
  float ly = (oc - (OUT_CHANNELS - 1) / 2.0) * PW_SLOT;
  float lz = (ch - (IN_CHANNELS - 1) / 2.0) * PW_CH_GAP;
  return addOrigin(pointwiseOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector pointwiseKernelCenter(int oc) {
  float lx = 0;
  float ly = (oc - (OUT_CHANNELS - 1) / 2.0) * PW_SLOT;
  float lz = 0;
  return addOrigin(pointwiseOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector spatialPosition(int ch, int row, int col) {
  float lx = (col - (MID_SIZE - 1) / 2.0) * MID_STEP;
  float ly = (row - (MID_SIZE - 1) / 2.0) * MID_STEP;
  float lz = (ch - (IN_CHANNELS - 1) / 2.0) * MID_LAYER_GAP;
  return addOrigin(spatialOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector finalPosition(int oc, int row, int col) {
  float lx = (col - (MID_SIZE - 1) / 2.0) * OUT_STEP;
  float ly = (row - (MID_SIZE - 1) / 2.0) * OUT_STEP;
  float lz = (oc - (OUT_CHANNELS - 1) / 2.0) * OUT_LAYER_GAP;
  return addOrigin(finalOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector patchCenter(int ch) {
  float centerRow = outRow;
  float centerCol = outCol;
  float lx = (centerCol - (MID_SIZE - 1) / 2.0) * MID_STEP;
  float ly = (centerRow - (MID_SIZE - 1) / 2.0) * MID_STEP;
  float lz = (ch - (IN_CHANNELS - 1) / 2.0) * IN_LAYER_GAP;
  return addOrigin(inputOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector depthwiseKernelCenter(int ch) {
  float lx = 0;
  float ly = 0;
  float lz = (ch - (IN_CHANNELS - 1) / 2.0) * KERNEL_LAYER_GAP;
  return addOrigin(kernelOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector spatialStackCenter() {
  float lx = (outCol - (MID_SIZE - 1) / 2.0) * MID_STEP;
  float ly = (outRow - (MID_SIZE - 1) / 2.0) * MID_STEP;
  float lz = 0;
  return addOrigin(spatialOrigin, rotateLocalYCCW(lx, ly, lz));
}

PVector interpolate(PVector a, PVector b, float t) {
  return new PVector(
    lerp(a.x, b.x, t),
    lerp(a.y, b.y, t),
    lerp(a.z, b.z, t)
  );
}

float easeInOut(float t) {
  t = constrain(t, 0, 1);
  return t < 0.5 ? 2.0 * t * t : 1.0 - pow(-2.0 * t + 2.0, 2) / 2.0;
}

color valueColor(float t) {
  t = constrain(t, 0, 1);
  if (t < 0.5) {
    float s = map(t, 0, 0.5, 0, 1);
    return lerpColor(color(34, 74, 196), color(73, 224, 255), s);
  } else {
    float s = map(t, 0.5, 1.0, 0, 1);
    return lerpColor(color(73, 224, 255), color(255, 222, 76), s);
  }
}

color kernelColor(float rawValue, float normValue) {
  if (rawValue >= 0) {
    return lerpColor(color(26, 116, 160), color(90, 228, 255), normValue);
  } else {
    return lerpColor(color(140, 40, 88), color(255, 110, 150), 1.0 - normValue);
  }
}

void drawHUD() {
  hint(DISABLE_DEPTH_TEST);
  camera();
  noLights();

  noStroke();
  fill(8, 12, 17, 232);
  rect(0, 0, width, 108);
  stroke(94, 110, 124, 80);
  line(0, 108, width, 108);

  fill(245);
  textSize(24);
  text("Depthwise separable convolution - 6 output channels", 28, 35);

  fill(175, 189, 200);
  textSize(13);
  text("Depthwise spatial filtering (6 kernels) then channel combining with 6 pointwise kernels to 20 x 20 x 6", 29, 62);

  drawMetric(640, 20, "INPUT", "20 x 20 x 6");
  drawMetric(800, 20, "DEPTHWISE", "6 x (3 x 3)");
  drawMetric(1000, 20, "POINTWISE", "6 x (1 x 1 x 6)");
  drawMetric(1230, 20, "OUTPUT", "20 x 20 x 6");

  fill(240, 240, 245);
  textSize(15);
  text("Input", 95, 138);
  text("Depthwise kernels", 300, 138);
  text("Spatial Filtering", width / 2.0 - 130, 138);
  text("Combining kernels", width / 2.0 + 120, 138);
  text("Channel Combining output", width - 300, 138);

  fill(153, 168, 180);
  textSize(11);
  text("padding = 1, stride = 1", 300, 156);
  text("6 intermediate planes (20 x 20)", width / 2.0 - 130, 156);
  text("6 pointwise 1x1x6 kernels", width / 2.0 + 120, 156);
  text("6 output planes (20 x 20)", width - 300, 156);

  noStroke();
  fill(8, 12, 17, 224);
  rect(20, height - 150, 700, 126, 7);
  stroke(94, 110, 124, 80);
  noFill();
  rect(20, height - 150, 700, 126, 7);

  fill(activeColor);
  textSize(13);
  text("CURRENT POSITION", 38, height - 122);
  fill(244);
  textSize(20);
  text("row " + outRow + ", column " + outCol, 38, height - 94);

  fill(168, 183, 194);
  textSize(13);
  text("Phase: " + phaseLabel(), 38, height - 67);
  if (phase == 0 || phase == 1) {
    text("Active depthwise kernel: input channel " + depthChannel + " / 5", 38, height - 44);
  } else {
    text("Active combining kernel: output channel " + outputChannel + " / 5", 38, height - 44);
  }
  text("6 combining kernels -> 6 output channels", 400, height - 44);

  noStroke();
  fill(8, 12, 17, 224);
  rect(width - 500, height - 150, 480, 126, 7);
  stroke(94, 110, 124, 80);
  noFill();
  rect(width - 500, height - 150, 480, 126, 7);

  fill(244);
  textSize(13);
  text("SPACE  play/pause", width - 480, height - 122);
  text("RIGHT  complete current position", width - 480, height - 98);
  text("R  reset     N  new data", width - 480, height - 74);
  text("V  values    [ ]  speed", width - 480, height - 50);
  text("Drag  orbit     Wheel  zoom     1/2/3  views", width - 245, height - 98, 220, 70);

  fill(playing ? activeColor : mutedColor);
  noStroke();
  ellipse(width - 42, 40, 10, 10);
  fill(playing ? 230 : 160);
  textAlign(RIGHT, CENTER);
  textSize(12);
  text(phase == 4 ? "DONE" : (playing ? "PLAYING" : "PAUSED"), width - 57, 40);
  textAlign(LEFT, BASELINE);

  if (showValues) drawValueOverlay();

  hint(ENABLE_DEPTH_TEST);
}

String phaseLabel() {
  if (phase == 0) return "Inspecting one channel patch and its matching 3 x 3 kernel";
  if (phase == 1) return "Computing one intermediate depthwise output voxel";
  if (phase == 2) return "Inspecting the six intermediate channels + combining kernel " + outputChannel;
  if (phase == 3) return "Computing one final combined output voxel (channel " + outputChannel + ")";
  return "Completed";
}

void drawMetric(float x, float y, String label, String value) {
  fill(126, 143, 156);
  textSize(11);
  text(label, x, y + 12);
  fill(240);
  textSize(16);
  text(value, x, y + 38);
}

void drawValueOverlay() {
  float panelX = width / 2.0 - 290;
  float panelY = 170;

  noStroke();
  fill(7, 10, 14, 236);
  rect(panelX, panelY, 580, 150, 7);
  stroke(94, 110, 124, 90);
  noFill();
  rect(panelX, panelY, 580, 150, 7);

  fill(242);
  textSize(13);
  text("Current numeric contributions", panelX + 16, panelY + 24);

  fill(activeColor);
  text("Combining into output channel " + outputChannel, panelX + 16, panelY + 48);

  float running = 0;
  for (int ch = 0; ch < IN_CHANNELS; ch++) {
    float contribution = spatialOut[ch][outRow][outCol] * pointwiseWeights[outputChannel][ch];
    running += contribution;
    fill(contribution >= 0 ? positiveColor : negativeColor);
    text("C" + ch + ": " + nf(contribution, 1, 3), panelX + 16 + (ch % 3) * 175, panelY + 78 + (ch / 3) * 24);
  }

  fill(activeColor);
  text("Final output (channel " + outputChannel + ") = " + nf(running, 1, 3), panelX + 16, panelY + 134);
}

void keyPressed() {
  if (key == ' ') {
    if (phase != 4) playing = !playing;
  } else if (key == 'r' || key == 'R') {
    resetAnimation();
  } else if (key == 'n' || key == 'N') {
    randomizeData();
    resetAnimation();
  } else if (key == 'v' || key == 'V') {
    showValues = !showValues;
  } else if (key == '[') {
    animationSpeed = max(0.25, animationSpeed - 0.25);
  } else if (key == ']') {
    animationSpeed = min(8.0, animationSpeed + 0.25);
  } else if (key == '1') {
    yaw = -0.47;
    pitch = 0.40;
    zoomAmount = 0.80;
  } else if (key == '2') {
    yaw = 0;
    pitch = 0;
    zoomAmount = 0.78;
  } else if (key == '3') {
    yaw = 0;
    pitch = HALF_PI - 0.03;
    zoomAmount = 0.74;
  }

  if (key == CODED && keyCode == RIGHT && phase != 4) {
    completeCurrentPosition();
  }
}

void mousePressed() {
  previousMouseX = mouseX;
  previousMouseY = mouseY;
}

void mouseDragged() {
  float dx = mouseX - previousMouseX;
  float dy = mouseY - previousMouseY;
  yaw += dx * 0.008;
  pitch = constrain(pitch + dy * 0.008, -1.35, 1.35);
  previousMouseX = mouseX;
  previousMouseY = mouseY;
}

void mouseWheel(MouseEvent event) {
  zoomAmount = constrain(zoomAmount - event.getCount() * 0.06, 0.45, 1.75);
}
