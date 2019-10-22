package org.tensorflow.lite.examples.detection.tflite;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;
import org.json.JSONException;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class YoloDetector {

    private final Interpreter tfLite;
    RenderScript rs;

    private final List<Double> anchors;
    private ByteBuffer input;
    private int[] intValues;
    private float[][][][] output;

    private final int INP_IMG_WIDTH;
    private final int INP_IMG_HEIGHT;
    private final int NUM_BOXES_PER_BLOCK;
    private final int NUM_CLASSES;
    private final int gridWidth;
    private final int gridHeight;
    private final int blockSize;

    private final int MAX_RESULTS;
    private final double THRESHOLD;
    private final double OVERLAP_THRESHOLD;


    /** Tag for the {@link Log}. */
    private static final String TAG = "YoloDetector";

    YoloDetector(RenderScript rs, ByteBuffer modalData, Map meta) throws IOException, JSONException {

        tfLite = new Interpreter(modalData);

        /** Initialize Input Buffer based on meta as Byte Buffer**/
        blockSize = (int) meta.get("blockSize");

        Map net = (Map) meta.get("net");
        INP_IMG_WIDTH = (int) net.get("width");
        INP_IMG_HEIGHT = (int) net.get("height");
        NUM_CLASSES = (int) meta.get("classes");
        NUM_BOXES_PER_BLOCK = (int) meta.get("num");
        MAX_RESULTS = (int) meta.get("max_result");
        THRESHOLD= (double) meta.get("threshold");
        OVERLAP_THRESHOLD =  (double) meta.get("overlap_threshold");

        input = ByteBuffer.allocateDirect(
                4 * (int) net.get("batch") * INP_IMG_WIDTH * INP_IMG_HEIGHT * (int) net.get("channels"));
        input.order(ByteOrder.nativeOrder());
        intValues = new int[INP_IMG_WIDTH * INP_IMG_HEIGHT];

        gridWidth = INP_IMG_WIDTH / blockSize;
        gridHeight = INP_IMG_HEIGHT / blockSize;
        System.out.println(gridWidth);
        System.out.println(gridHeight);
        List<Integer> outputDim = (List<Integer>) meta.get("out_size");
        output = new float[1][outputDim.get(0)][outputDim.get(1)][outputDim.get(2)];

        /** Get Meta Data for post processing **/
        anchors = (List<Double>) meta.get("anchors");
        this.rs=rs;
        Log.d(TAG, "Created a Tensorflow Lite Yolo Detector.");
    }



    public List<Map<String, Object>> detect(HashMap image){
        if (tfLite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            return null;
        }

        convertBitmapToByteBuffer(getBitmap(image));

        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();

        tfLite.run(input, output);

        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        return postProcess(output[0]);

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
                    if (confidenceInClass >  THRESHOLD) {
                        Map<String, Object> prediction = new HashMap<>();
                        prediction.put("classIndex",detectedClass);
                        prediction.put("confidence",confidenceInClass);
                        final float xPos = (x + expit(output[y][x][offset + 0])) * blockSize;
                        final float yPos = (y + expit(output[y][x][offset + 1])) * blockSize;

                        final float w = (float) (Math.exp(output[y][x][offset + 2]) * anchors.get(2 * b + 0)) * blockSize;
                        final float h = (float) (Math.exp(output[y][x][offset + 3]) * anchors.get(2 * b + 1)) * blockSize;

                        Map<String, Float> rectF = new HashMap<>();
                        rectF.put("left", Math.max(0, xPos - w / 2)); // left should have lower value as right
                        rectF.put("top", Math.max(0, yPos - h / 2));  // top should have lower value as bottom
                        rectF.put("right",Math.min(INP_IMG_WIDTH- 1, xPos + w / 2));
                        rectF.put("bottom",Math.min(INP_IMG_HEIGHT - 1, yPos + h / 2));
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

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (input == null) {
            return;
        }
        input.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < INP_IMG_WIDTH; ++i) {
            for (int j = 0; j < INP_IMG_HEIGHT; ++j) {
                final int val = intValues[pixel++];
                int IMAGE_MEAN = 128;
                float IMAGE_STD = 128.0f;
                input.putFloat((((val >> 16) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD);
                input.putFloat((((val >> 8) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD);
                input.putFloat((((val) & 0xFF)- IMAGE_MEAN)/ IMAGE_STD);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    public Bitmap getBitmap(HashMap image){
        Bitmap bitmap = Bitmap.createScaledBitmap(yuv420toBitMap(image), 416, 416, true);
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postRotate((Integer)image.get("rotation"));
        return Bitmap.createBitmap(bitmap   , 0, 0, width, height, matrix, true);
    }

    public Bitmap yuv420toBitMap(final HashMap image) {
        int w = (int) image.get("width");
        int h = (int) image.get("height");
        ArrayList<Map> planes = (ArrayList) image.get("planes");

        byte[] data = yuv420toNV21(w, h, planes);

        Bitmap bitmap = Bitmap.createBitmap(w, h,   Bitmap.Config.ARGB_8888);
        ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic =     ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

        Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(data.length);
        Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);
        in.copyFrom(data);

        Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(w).setY(h);
        Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);

        out.copyTo(bitmap);
        return bitmap;
    }

    public byte[] yuv420toNV21(int width,int height, ArrayList<Map> planes){
        byte[] yBytes = (byte[]) planes.get(0).get("bytes"),
                uBytes= (byte[]) planes.get(1).get("bytes"),
                vBytes= (byte[]) planes.get(2).get("bytes");
        final int color_pixel_stride =(int) planes.get(1).get("bytesPerPixel");

        ByteArrayOutputStream outputbytes = new ByteArrayOutputStream();
        try {
            outputbytes.write(yBytes);
            outputbytes.write(vBytes);
            outputbytes.write(uBytes);
        } catch (IOException e) {
            e.printStackTrace();
        }

        byte[] data = outputbytes.toByteArray();
        final int y_size = yBytes.length;
        final int u_size = uBytes.length;
        final int data_offset = width * height;
        for (int i = 0; i < y_size; i++) {
            data[i] = (byte) (yBytes[i] & 255);
        }
        for (int i = 0; i < u_size / color_pixel_stride; i++) {
            data[data_offset + 2 * i] = vBytes[i * color_pixel_stride];
            data[data_offset + 2 * i + 1] = uBytes[i * color_pixel_stride];
        }
        return data;
    }


}