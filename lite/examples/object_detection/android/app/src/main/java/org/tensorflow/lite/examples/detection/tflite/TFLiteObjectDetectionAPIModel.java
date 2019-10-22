/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.gpu.GpuDelegate;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static final Logger LOGGER = new Logger();

  int blockSize = 32;
  int gridHeight = 13;
  int gridWidth = 13;
  int NUM_CLASSES = 80;
  float THRESHOLD = 0.3f;
  int MAX_RESULTS = 25;
  float OVERLAP_THRESHOLD = 0.7f;
  int NUM_BOXES_PER_BLOCK = 5;
  float[] anchors = new float[] {
    0.57273f,
    0.677385f,
    1.87446f,
    2.06253f,
    3.33843f,
    5.47434f,
    7.88282f,
    3.52778f,
    9.77052f,
    9.16828f
  };

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private ByteBuffer imgData;

  private Interpreter tfLite;

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      Interpreter.Options tfliteOptions = new Interpreter.Options();
//      GpuDelegate gpuDelegate = new GpuDelegate();
//      tfliteOptions.addDelegate(gpuDelegate);
      MappedByteBuffer buff = loadModelFile(assetManager, modelFilename);
      d.tfLite = new Interpreter(buff, tfliteOptions);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][13][13][425];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    int pixel = 0;
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        final int val = intValues[pixel++];
        int IMAGE_MEAN = 128;
        float IMAGE_STD = 128.0f;
        imgData.putFloat((((val >> 16) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD);
        imgData.putFloat((((val >> 8) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD);
        imgData.putFloat((((val) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD);
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    outputLocations = new float[1][13][13][425];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    long start = System.currentTimeMillis();
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    System.out.println("Processed in ms: " + (System.currentTimeMillis() - start));
    Trace.endSection();

    // https://medium.com/@y1017c121y/how-does-yolov2-work-daaaa967c5f7
    final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
    List<Map<String, Object>> detections = postProcess(outputLocations[0]);

    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  public List<Map<String, Object>> postProcess(final float[][][] output) {
    // Find the best detections.
    PriorityQueue<Map<String, Object>> priorityQueue = new PriorityQueue<>(1, new PredictionComparator());

    for (int y = 0; y < gridHeight; ++y) {
      for (int x = 0; x < gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
          final int offset = (NUM_CLASSES + 5) * b;

          final float confidence = expit(output[y][x][offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = output[x][y][offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
//          System.out.println("" + confidenceInClass);
          if (confidenceInClass >  THRESHOLD) {
            Map<String, Object> prediction = new HashMap<>();
            prediction.put("classIndex",detectedClass);
            prediction.put("confidence",confidenceInClass);
            final float xPos = (x + expit(output[y][x][offset + 0])) * blockSize;
            final float yPos = (y + expit(output[y][x][offset + 1])) * blockSize;

            final float w = (float) (Math.exp(output[y][x][offset + 2]) * anchors[2 * b + 0]) * blockSize;
            final float h = (float) (Math.exp(output[y][x][offset + 3]) * anchors[2 * b + 1]) * blockSize;

            Map<String, Float> rectF = new HashMap<>();
            rectF.put("left", Math.max(0, xPos - w / 2)); // left should have lower value as right
            rectF.put("top", Math.max(0, yPos - h / 2));  // top should have lower value as bottom
            rectF.put("right",Math.min(640- 1, xPos + w / 2));
            rectF.put("bottom",Math.min(480 - 1, yPos + h / 2));
            prediction.put("rect",rectF);
            priorityQueue.add(prediction);
          }
        }
      }
    }

    final List<Map<String, Object>> predictions = new ArrayList<>();
    Map<String, Object> bestPrediction = priorityQueue.poll();
    predictions.add(bestPrediction);

    for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
      Map<String, Object> prediction = priorityQueue.poll();
      boolean overlaps = false;
      for (Map<String, Object> previousPrediction : predictions) {
        float intersectProportion = 0f;
        Map<String, Float> primary = (Map<String, Float>) previousPrediction.get("rect");
        Map<String, Float> secondary =  (Map<String, Float>) prediction.get("rect");
        if (primary.get("left") < secondary.get("right") && primary.get("right") > secondary.get("left")
                && primary.get("top") < secondary.get("bottom") && primary.get("bottom") > secondary.get("top")) {
          float intersection = Math.max(0, Math.min(primary.get("right"), secondary.get("right")) - Math.max(primary.get("left"), secondary.get("left"))) *
                  Math.max(0, Math.min(primary.get("bottom"), secondary.get("bottom")) - Math.max(primary.get("top"), secondary.get("top")));

          float main = Math.abs(primary.get("right") - primary.get("left")) * Math.abs(primary.get("bottom") - primary.get("top"));
          intersectProportion= intersection / main;
        }

        overlaps = overlaps || (intersectProportion > OVERLAP_THRESHOLD);
      }

      if (!overlaps) {
        predictions.add(prediction);
      }
    }
    return predictions;
  }

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  private class PredictionComparator implements Comparator<Map<String, Object>> {
    @Override
    public int compare(final Map<String, Object> prediction1, final Map<String, Object> prediction2) {
      return Float.compare((float)prediction2.get("confidence"), (float)prediction1.get("confidence"));
    }
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}
