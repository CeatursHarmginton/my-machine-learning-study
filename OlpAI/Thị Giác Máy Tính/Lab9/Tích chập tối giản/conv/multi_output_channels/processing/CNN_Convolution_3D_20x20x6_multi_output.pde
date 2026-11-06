import processing.event.MouseEvent;

/*
  3D CNN convolution visualization - MULTIPLE OUTPUT CHANNELS

  Input:   20 x 20 x 6
  Kernels: 6 separate kernels, each 3 x 3 x 6
  Stride:  1
  Padding: 0 (valid)
  Output:  18 x 18 x 6

  This is the multi-output-channel version of the single-output sketch.
  A standard convolution layer produces one output channel per kernel.
  To obtain 6 output channels we therefore use 6 independent 3 x 3 x 6
  kernels. Each kernel slides over the whole 20 x 20 x 6 input and writes
  into its own 18 x 18 output plane.

  Visual approach inspired by the layered voxel animation style of:
  https://github.com/okdalto/CNN-visualization

  Tensor planes, kernels, and output planes are rotated 90 degrees
  counterclockwise around the Y axis.
*/

final int INPUT_SIZE = 20;
final int INPUT_CHANNELS = 6;
final int KERNEL_SIZE = 3;
final int STRIDE = 1;
final int OUTPUT_SIZE = 18;
final int OUTPUT_CHANNELS = 6;

final float ROTATE_Y_CCW = HALF_PI;

final float INPUT_STEP = 6.6;
final float INPUT_LAYER_GAP = 16.5;
final float VOXEL_SIZE = 5.3;
final float VOXEL_DEPTH = 3.6;

final float OUTPUT_STEP = 6.6;
final float OUT_LAYER_GAP = 15.5;

// kernel stack layout: 6 kernels stacked vertically, each a 3 x 3 x 6 slab
final float KERNEL_STEP = 6.2;       // spacing inside one kernel (x/y)
final float KERNEL_CH_GAP = 9.0;     // spacing between the 6 channels of one kernel
final float KERNEL_SLOT = 41.0;      // vertical spacing between the 6 kernels
final float KERNEL_VOXEL_SIZE = 5.4;

final float MOVING_VOXEL_SIZE = 5.4;

float[][][] inputTensor = new float[INPUT_CHANNELS][INPUT_SIZE][INPUT_SIZE];
float[][][][] kernel = new float[OUTPUT_CHANNELS][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
float[][][] output = new float[OUTPUT_CHANNELS][OUTPUT_SIZE][OUTPUT_SIZE];
float[][][] outputNormalized = new float[OUTPUT_CHANNELS][OUTPUT_SIZE][OUTPUT_SIZE];
float[][][][] kernelNorm = new float[OUTPUT_CHANNELS][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
boolean[][][] revealed = new boolean[OUTPUT_CHANNELS][OUTPUT_SIZE][OUTPUT_SIZE];

PVector inputOrigin = new PVector(-500, 0, -20);
PVector kernelOrigin = new PVector(-30, 0, 0);
PVector outputOrigin = new PVector(470, 0, 10);

int outCol = 0;
int outRow = 0;
int outChannel = 0;
int phase = 0;          // 0: inspect, 1: transfer, 2: settle
float phaseTime = 0;
boolean playing = true;
boolean showValues = false;
float animationSpeed = 8.0;

float yaw = -0.52;
float pitch = 0.40;
float zoomAmount = 0.82;
float previousMouseX;
float previousMouseY;
int lastFrameMillis;

color backgroundColor;
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
  surface.setTitle("3D CNN Convolution: 20x20x6 to 18x18x6 (6 kernels)");
  frameRate(60);

  backgroundColor = color(4, 6, 9);
  wireColor = color(225, 232, 238);
  mutedColor = color(76, 88, 98);
  activeColor = color(255, 190, 60);
  positiveColor = color(104, 225, 255);
  negativeColor = color(255, 112, 150);

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

  background(backgroundColor);
  perspective(PI / 3.1, float(width) / float(height), 5, 6000);

  ambientLight(72, 72, 78);
  directionalLight(210, 215, 225, -0.25, 0.45, -1);

  pushMatrix();
  translate(width * 0.50, height * 0.56, 0);
  scale(zoomAmount);
  rotateX(pitch);
  rotateY(yaw);

  drawFloorGrid();
  drawTensorFrames();
  drawInputTensor();
  drawKernelStack();
  drawOutputTensor();
  drawTransferPath();
  drawMovingKernelVolume();

  popMatrix();

  drawHUD();
}

void randomizeData() {
  noiseSeed((int)random(100000));

  for (int c = 0; c < INPUT_CHANNELS; c++) {
    float channelOffset = c * 17.0;
    for (int r = 0; r < INPUT_SIZE; r++) {
      for (int col = 0; col < INPUT_SIZE; col++) {
        float n1 = noise(col * 0.16, r * 0.16, channelOffset);
        float n2 = noise(col * 0.36 + 20, r * 0.36 + 20, channelOffset + 8);
        inputTensor[c][r][col] = constrain(0.72 * n1 + 0.28 * n2, 0, 1);
      }
    }
  }

  for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
    for (int c = 0; c < INPUT_CHANNELS; c++) {
      for (int kr = 0; kr < KERNEL_SIZE; kr++) {
        for (int kc = 0; kc < KERNEL_SIZE; kc++) {
          float radial = 1.0 - dist(kc, kr, 1, 1) / 1.8;
          float channelSign = ((oc + c) % 2 == 0) ? 1.0 : -1.0;
          kernel[oc][c][kr][kc] = radial * channelSign * 0.45 + random(-0.55, 0.55);
        }
      }
    }
  }

  calculateEntireOutput();
  normalizeKernels();
}

void calculateEntireOutput() {
  for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
    float minValue = Float.MAX_VALUE;
    float maxValue = -Float.MAX_VALUE;

    for (int r = 0; r < OUTPUT_SIZE; r++) {
      for (int col = 0; col < OUTPUT_SIZE; col++) {
        output[oc][r][col] = convolutionAt(oc, r, col);
        minValue = min(minValue, output[oc][r][col]);
        maxValue = max(maxValue, output[oc][r][col]);
      }
    }

    float range = max(maxValue - minValue, 0.0001);
    for (int r = 0; r < OUTPUT_SIZE; r++) {
      for (int col = 0; col < OUTPUT_SIZE; col++) {
        outputNormalized[oc][r][col] = (output[oc][r][col] - minValue) / range;
      }
    }
  }
}

float convolutionAt(int oc, int outputRow, int outputCol) {
  float sum = 0;
  for (int c = 0; c < INPUT_CHANNELS; c++) {
    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
      for (int kc = 0; kc < KERNEL_SIZE; kc++) {
        int inputRow = outputRow * STRIDE + kr;
        int inputCol = outputCol * STRIDE + kc;
        sum += inputTensor[c][inputRow][inputCol] * kernel[oc][c][kr][kc];
      }
    }
  }
  return sum;
}

void normalizeKernels() {
  float absMax = 0.0001;
  for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
    for (int c = 0; c < INPUT_CHANNELS; c++) {
      for (int kr = 0; kr < KERNEL_SIZE; kr++) {
        for (int kc = 0; kc < KERNEL_SIZE; kc++) {
          absMax = max(absMax, abs(kernel[oc][c][kr][kc]));
        }
      }
    }
  }
  for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
    for (int c = 0; c < INPUT_CHANNELS; c++) {
      for (int kr = 0; kr < KERNEL_SIZE; kr++) {
        for (int kc = 0; kc < KERNEL_SIZE; kc++) {
          kernelNorm[oc][c][kr][kc] = map(kernel[oc][c][kr][kc], -absMax, absMax, 0, 1);
        }
      }
    }
  }
}

void resetAnimation() {
  for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
    for (int r = 0; r < OUTPUT_SIZE; r++) {
      for (int col = 0; col < OUTPUT_SIZE; col++) {
        revealed[oc][r][col] = false;
      }
    }
  }
  outCol = 0;
  outRow = 0;
  outChannel = 0;
  phase = 0;
  phaseTime = 0;
}

void updateAnimation(float dt) {
  if (!playing) return;

  phaseTime += dt * animationSpeed;

  if (phase == 0 && phaseTime >= 0.45) {
    phase = 1;
    phaseTime = 0;
  } else if (phase == 1 && phaseTime >= 1.10) {
    revealed[outChannel][outRow][outCol] = true;
    phase = 2;
    phaseTime = 0;
  } else if (phase == 2 && phaseTime >= 0.16) {
    advanceCursor();
    phase = 0;
    phaseTime = 0;
  }
}

void advanceCursor() {
  outChannel++;
  if (outChannel < OUTPUT_CHANNELS) {
    return;
  }
  outChannel = 0;

  outCol++;
  if (outCol >= OUTPUT_SIZE) {
    outCol = 0;
    outRow++;
  }
  if (outRow >= OUTPUT_SIZE) {
    outRow = 0;
    for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
      for (int r = 0; r < OUTPUT_SIZE; r++) {
        for (int col = 0; col < OUTPUT_SIZE; col++) {
          revealed[oc][r][col] = false;
        }
      }
    }
  }
}

void completeCurrentStep() {
  revealed[outChannel][outRow][outCol] = true;
  advanceCursor();
  phase = 0;
  phaseTime = 0;
}

void drawInputTensor() {
  for (int c = 0; c < INPUT_CHANNELS; c++) {
    for (int r = 0; r < INPUT_SIZE; r++) {
      for (int col = 0; col < INPUT_SIZE; col++) {
        boolean selected = col >= outCol && col < outCol + KERNEL_SIZE &&
                           r >= outRow && r < outRow + KERNEL_SIZE;

        PVector p = inputPosition(c, r, col);
        float value = inputTensor[c][r][col];
        color voxelColor = valueColor(value, 118);

        pushMatrix();
        translate(p.x, p.y, p.z);

        if (selected) {
          stroke(activeColor, phase == 1 ? 135 : 255);
          strokeWeight(1.25);
          fill(activeColor, phase == 1 ? 32 : 110);
        } else {
          stroke(wireColor, 42);
          strokeWeight(0.55);
          fill(voxelColor);
        }

        box(VOXEL_SIZE, VOXEL_SIZE, VOXEL_DEPTH);
        popMatrix();
      }
    }
  }
}

void drawKernelStack() {
  for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
    boolean activeKernel = oc == outChannel;

    for (int c = 0; c < INPUT_CHANNELS; c++) {
      for (int kr = 0; kr < KERNEL_SIZE; kr++) {
        for (int kc = 0; kc < KERNEL_SIZE; kc++) {
          PVector p = kernelPosition(oc, c, kr, kc);
          color vc = kernelColor(kernel[oc][c][kr][kc], kernelNorm[oc][c][kr][kc]);

          pushMatrix();
          translate(p.x, p.y, p.z);
          stroke(activeKernel ? activeColor : wireColor, activeKernel ? 210 : 55);
          strokeWeight(activeKernel ? 1.1 : 0.7);
          fill(vc, activeKernel ? 225 : 120);
          box(KERNEL_VOXEL_SIZE, KERNEL_VOXEL_SIZE, 3.6);
          popMatrix();
        }
      }
    }
  }
}

color valueColor(float value, float alphaValue) {
  float v = constrain(value, 0, 1);

  color low = color(42, 76, 170);
  color middle = color(96, 220, 238);
  color high = color(255, 196, 70);

  color mapped;
  if (v < 0.5) {
    mapped = lerpColor(low, middle, v * 2.0);
  } else {
    mapped = lerpColor(middle, high, (v - 0.5) * 2.0);
  }

  return color(red(mapped), green(mapped), blue(mapped), alphaValue);
}

color kernelColor(float rawValue, float normValue) {
  if (rawValue >= 0) {
    return lerpColor(color(26, 116, 160), color(90, 228, 255), normValue);
  } else {
    return lerpColor(color(140, 40, 88), color(255, 110, 150), 1.0 - normValue);
  }
}

void drawOutputTensor() {
  for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
    for (int r = 0; r < OUTPUT_SIZE; r++) {
      for (int col = 0; col < OUTPUT_SIZE; col++) {
        PVector p = outputPosition(oc, r, col);
        boolean current = oc == outChannel && r == outRow && col == outCol;

        pushMatrix();
        translate(p.x, p.y, p.z);

        if (revealed[oc][r][col]) {
          float v = outputNormalized[oc][r][col];
          stroke(wireColor, 140);
          strokeWeight(0.7);
          fill(valueColor(v, 215));
          box(VOXEL_SIZE, VOXEL_SIZE, VOXEL_DEPTH);
        } else if (current) {
          stroke(activeColor, 255);
          strokeWeight(1.5);
          fill(activeColor, phase == 2 ? 150 : 22);
          box(VOXEL_SIZE, VOXEL_SIZE, VOXEL_DEPTH);
        } else {
          stroke(mutedColor, 80);
          strokeWeight(0.45);
          noFill();
          box(VOXEL_SIZE, VOXEL_SIZE, VOXEL_DEPTH);
        }

        popMatrix();
      }
    }
  }
}

void drawMovingKernelVolume() {
  if (phase != 1) return;

  float progress = constrain(phaseTime / 1.10, 0, 1);
  float split = 0.53;

  for (int c = 0; c < INPUT_CHANNELS; c++) {
    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
      for (int kc = 0; kc < KERNEL_SIZE; kc++) {
        PVector source = inputPosition(c, outRow + kr, outCol + kc);
        PVector weightPos = kernelPosition(outChannel, c, kr, kc);
        PVector target = outputPosition(outChannel, outRow, outCol);

        PVector position;
        float localProgress;

        if (progress < split) {
          localProgress = easeInOut(progress / split);
          position = interpolate(source, weightPos, localProgress);
        } else {
          localProgress = easeInOut((progress - split) / (1.0 - split));
          position = interpolate(weightPos, target, localProgress);
        }

        float product = inputTensor[c][outRow + kr][outCol + kc] * kernel[outChannel][c][kr][kc];
        float magnitude = constrain(abs(product) * 2.8, 0, 1);
        color productColor = product >= 0 ? positiveColor : negativeColor;

        pushMatrix();
        translate(position.x, position.y, position.z);
        stroke(productColor, 220);
        strokeWeight(1.1);
        fill(productColor, 55 + 135 * magnitude);
        box(MOVING_VOXEL_SIZE, MOVING_VOXEL_SIZE, VOXEL_DEPTH);
        popMatrix();
      }
    }
  }
}

void drawTransferPath() {
  PVector source = selectedVolumeCenter();
  PVector activeKernelCenter = kernelStackCenter(outChannel);
  PVector target = outputPosition(outChannel, outRow, outCol);

  stroke(activeColor, phase == 1 ? 150 : 55);
  strokeWeight(1.0);
  noFill();
  line(source.x, source.y, source.z, activeKernelCenter.x, activeKernelCenter.y, activeKernelCenter.z);
  line(activeKernelCenter.x, activeKernelCenter.y, activeKernelCenter.z, target.x, target.y, target.z);

  pushMatrix();
  translate(activeKernelCenter.x, activeKernelCenter.y, activeKernelCenter.z);
  rotateY(ROTATE_Y_CCW);
  stroke(activeColor, phase == 1 ? 180 : 55);
  noFill();
  box((KERNEL_SIZE - 1) * KERNEL_STEP + 12,
      (KERNEL_SIZE - 1) * KERNEL_STEP + 12,
      (INPUT_CHANNELS - 1) * KERNEL_CH_GAP + 10);
  popMatrix();
}

void drawTensorFrames() {
  pushMatrix();
  translate(inputOrigin.x, inputOrigin.y, inputOrigin.z);
  rotateY(ROTATE_Y_CCW);
  stroke(wireColor, 90);
  strokeWeight(0.9);
  noFill();
  box((INPUT_SIZE - 1) * INPUT_STEP + 10,
      (INPUT_SIZE - 1) * INPUT_STEP + 10,
      (INPUT_CHANNELS - 1) * INPUT_LAYER_GAP + 9);
  popMatrix();

  pushMatrix();
  translate(outputOrigin.x, outputOrigin.y, outputOrigin.z);
  rotateY(ROTATE_Y_CCW);
  stroke(wireColor, 95);
  strokeWeight(0.9);
  noFill();
  box((OUTPUT_SIZE - 1) * OUTPUT_STEP + 10,
      (OUTPUT_SIZE - 1) * OUTPUT_STEP + 10,
      (OUTPUT_CHANNELS - 1) * OUT_LAYER_GAP + 9);
  popMatrix();
}

void drawFloorGrid() {
  stroke(95, 110, 122, 28);
  strokeWeight(0.6);
  float y = 150;
  for (int i = -11; i <= 11; i++) {
    line(i * 70, y, -320, i * 70, y, 320);
    line(-720, y, i * 34, 720, y, i * 34);
  }
}

PVector rotateLocalYCCW(float x, float y, float z) {
  return new PVector(z, y, -x);
}

PVector addOrigin(PVector origin, PVector local) {
  return new PVector(origin.x + local.x, origin.y + local.y, origin.z + local.z);
}

PVector inputPosition(int channel, int row, int col) {
  float localX = (col - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float localY = (row - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float localZ = (channel - (INPUT_CHANNELS - 1) / 2.0) * INPUT_LAYER_GAP;
  return addOrigin(inputOrigin, rotateLocalYCCW(localX, localY, localZ));
}

PVector kernelPosition(int oc, int channel, int kernelRow, int kernelCol) {
  float localX = (kernelCol - (KERNEL_SIZE - 1) / 2.0) * KERNEL_STEP;
  float localY = (oc - (OUTPUT_CHANNELS - 1) / 2.0) * KERNEL_SLOT
               + (kernelRow - (KERNEL_SIZE - 1) / 2.0) * KERNEL_STEP;
  float localZ = (channel - (INPUT_CHANNELS - 1) / 2.0) * KERNEL_CH_GAP;
  return addOrigin(kernelOrigin, rotateLocalYCCW(localX, localY, localZ));
}

PVector kernelStackCenter(int oc) {
  float localX = 0;
  float localY = (oc - (OUTPUT_CHANNELS - 1) / 2.0) * KERNEL_SLOT;
  float localZ = 0;
  return addOrigin(kernelOrigin, rotateLocalYCCW(localX, localY, localZ));
}

PVector outputPosition(int channel, int row, int col) {
  float localX = (col - (OUTPUT_SIZE - 1) / 2.0) * OUTPUT_STEP;
  float localY = (row - (OUTPUT_SIZE - 1) / 2.0) * OUTPUT_STEP;
  float localZ = (channel - (OUTPUT_CHANNELS - 1) / 2.0) * OUT_LAYER_GAP;
  return addOrigin(outputOrigin, rotateLocalYCCW(localX, localY, localZ));
}

PVector selectedVolumeCenter() {
  float centerRow = outRow + (KERNEL_SIZE - 1) / 2.0;
  float centerCol = outCol + (KERNEL_SIZE - 1) / 2.0;
  float localX = (centerCol - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float localY = (centerRow - (INPUT_SIZE - 1) / 2.0) * INPUT_STEP;
  float localZ = 0;
  return addOrigin(inputOrigin, rotateLocalYCCW(localX, localY, localZ));
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

void drawHUD() {
  hint(DISABLE_DEPTH_TEST);
  camera();
  noLights();

  noStroke();
  fill(8, 12, 17, 232);
  rect(0, 0, width, 100);
  stroke(94, 110, 124, 80);
  line(0, 100, width, 100);

  fill(245);
  textSize(23);
  text("3D valid convolution - 6 output channels", 28, 34);

  fill(174, 188, 199);
  textSize(13);
  text("Six independent 3x3x6 kernels each sweep the 20x20x6 tensor to build one 18x18 output plane", 29, 60);

  drawMetric(600, 18, "INPUT", "20 x 20 x 6");
  drawMetric(770, 18, "KERNELS", "6 x (3 x 3 x 6)");
  drawMetric(990, 18, "STRIDE", "1");
  drawMetric(1110, 18, "PADDING", "0");
  drawMetric(1250, 18, "OUTPUT", "18 x 18 x 6");

  noStroke();
  fill(8, 12, 17, 224);
  rect(20, height - 150, 470, 126, 7);
  stroke(94, 110, 124, 80);
  noFill();
  rect(20, height - 150, 470, 126, 7);

  fill(activeColor);
  textSize(13);
  text("OUTPUT POSITION", 38, height - 122);
  fill(244);
  textSize(20);
  text("channel " + outChannel + "  |  row " + outRow + ", column " + outCol, 38, height - 93);

  fill(169, 184, 196);
  textSize(13);
  String phaseName = phase == 0 ? "Selecting 3x3 region + kernel " + outChannel
                   : phase == 1 ? "Multiplying and summing 54 values" : "Writing output value";
  text(phaseName, 38, height - 66);
  text("sum = " + nf(output[outChannel][outRow][outCol], 1, 3), 38, height - 44);
  text("O = (20 - 3) / 1 + 1 = 18   x 6 channels", 230, height - 44);

  noStroke();
  fill(8, 12, 17, 224);
  rect(width - 445, height - 150, 425, 126, 7);
  stroke(94, 110, 124, 80);
  noFill();
  rect(width - 445, height - 150, 425, 126, 7);

  fill(244);
  textSize(13);
  text("SPACE  play/pause", width - 425, height - 122);
  text("RIGHT  next kernel/output", width - 425, height - 98);
  text("R  reset     N  new data", width - 425, height - 74);
  text("Drag  orbit     Wheel  zoom", width - 425, height - 50);
  text("[ ]  speed     V  values     1/2/3  views", width - 225, height - 122, 200, 90);

  noStroke();
  fill(playing ? activeColor : mutedColor);
  ellipse(width - 40, 37, 10, 10);
  fill(playing ? 230 : 160);
  textSize(12);
  textAlign(RIGHT, CENTER);
  text(playing ? "PLAYING" : "PAUSED", width - 55, 37);
  textAlign(LEFT, BASELINE);

  if (showValues) {
    drawValueOverlay();
  }

  hint(ENABLE_DEPTH_TEST);
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
  float panelX = width / 2.0 - 170;
  float panelY = 112;
  noStroke();
  fill(7, 10, 14, 235);
  rect(panelX, panelY, 340, 128, 7);
  stroke(94, 110, 124, 90);
  noFill();
  rect(panelX, panelY, 340, 128, 7);

  fill(240);
  textSize(13);
  text("Channel contributions (output channel " + outChannel + ")", panelX + 16, panelY + 24);

  float running = 0;
  for (int c = 0; c < INPUT_CHANNELS; c++) {
    float contribution = 0;
    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
      for (int kc = 0; kc < KERNEL_SIZE; kc++) {
        contribution += inputTensor[c][outRow + kr][outCol + kc] * kernel[outChannel][c][kr][kc];
      }
    }
    running += contribution;
    fill(contribution >= 0 ? positiveColor : negativeColor);
    text("C" + c + ": " + nf(contribution, 1, 3), panelX + 16 + (c % 3) * 105, panelY + 54 + (c / 3) * 24);
  }

  fill(activeColor);
  text("Total (channel " + outChannel + "): " + nf(running, 1, 3), panelX + 16, panelY + 114);
}

void keyPressed() {
  if (key == ' ') {
    playing = !playing;
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
    animationSpeed = min(16.0, animationSpeed + 0.25);
  } else if (key == '1') {
    yaw = -0.52;
    pitch = 0.40;
    zoomAmount = 0.82;
  } else if (key == '2') {
    yaw = 0;
    pitch = 0;
    zoomAmount = 0.80;
  } else if (key == '3') {
    yaw = 0;
    pitch = HALF_PI - 0.03;
    zoomAmount = 0.78;
  }

  if (key == CODED && keyCode == RIGHT) {
    completeCurrentStep();
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
  zoomAmount = constrain(zoomAmount - event.getCount() * 0.06, 0.45, 1.9);
}
