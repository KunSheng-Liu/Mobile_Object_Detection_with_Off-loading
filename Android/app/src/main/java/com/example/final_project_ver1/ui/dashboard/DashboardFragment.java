package com.example.final_project_ver1.ui.dashboard;


import static android.content.ContentValues.TAG;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.renderscript.ScriptGroup;
import android.util.Log;
import android.util.Size;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.LifecycleOwner;

import com.example.final_project_ver1.OverlaySurface;
import com.example.final_project_ver1.R;
import com.example.final_project_ver1.databinding.FragmentDashboardBinding;
import com.google.common.util.concurrent.ListenableFuture;


import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TimerTask;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.InterpreterFactoryApi;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

public class DashboardFragment extends Fragment{

    androidx.camera.view.PreviewView previewView;

    SurfaceView surfaceView;
    SurfaceHolder sfhTrackHolder;

    Camera camera;
    Bitmap bitmapCamera;

    private Paint boxPaint;
    private Paint textBackgroundPaint;
    private Paint textPaint;
    private Rect bounds = new Rect();
    private int BOUNDING_RECT_TEXT_PADDING = 8;

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private FragmentDashboardBinding binding;

    private int[] tf_InputDim;
    private int[] tf_OutputDim;
    private static final String MODEL_FILE = "mobilenetv1.tflite";

    ObjectDetector objectDetector;
    List<Detection> results;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        binding = FragmentDashboardBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        initPaints();

        initialObjectDetector();

        previewView = binding.prevDashboard;
        surfaceView = binding.sfvDashboard;
        surfaceView.setZOrderOnTop(true);

        sfhTrackHolder = surfaceView.getHolder();
        sfhTrackHolder.setFormat(PixelFormat.TRANSPARENT);

        cameraProviderFuture = ProcessCameraProvider.getInstance(getContext());
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);

            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(getContext()));

        return root;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        // enable the following line if RGBA output is needed.
                        .setTargetResolution(new Size(640 , 480))
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        Executor cameraExecutor = Executors.newSingleThreadExecutor();
        imageAnalysis.setAnalyzer(cameraExecutor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                Log.d(TAG, "Image analyzer.");

                int imageRotation = imageProxy.getImageInfo().getRotationDegrees();
                Log.d (TAG, "Rotation: " + imageRotation);

                /**
                 * Convert imageProxy to RGB Bitmap
                 */
                if (bitmapCamera == null) {
                    // The image rotation and RGB image buffer are initialized only once
                    // the analyzer has started running
                    bitmapCamera = Bitmap.createBitmap(
                            imageProxy.getWidth(),
                            imageProxy.getHeight(),
                            Bitmap.Config.ARGB_8888
                    );
                }

                ByteBuffer rgb_buffer = imageProxy.getPlanes()[0].getBuffer();
                bitmapCamera.copyPixelsFromBuffer(rgb_buffer);

                Log.d (TAG, "Bitmap update");

                /**
                 * Data preprocessing to Tensor format
                 */
                ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                        .add(new ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new Rot90Op(-imageRotation / 90))
                        .build();

                TensorImage tensorImage = new TensorImage(DataType.UINT8);
                tensorImage.load(bitmapCamera);

                tensorImage = imageProcessor.process(tensorImage);
                Log.d(TAG, "tensorImage dim: " + tensorImage.getHeight() + " " + tensorImage.getWidth());

                TensorBuffer probabilityBuffer =
                        TensorBuffer.createFixedSize (new int[]{1, 1001}, DataType.UINT8);

                /**
                 * Start detect
                 */
//                detectByInterpreter (tensorImage, probabilityBuffer);
                detectByObjectDetector (tensorImage);

                Canvas canvas = null;
                try {
                    canvas = sfhTrackHolder.lockCanvas();
                    Paint clearPaint = new Paint();
                    clearPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
                    canvas.drawRect(0, 0, canvas.getWidth(), canvas.getHeight(), clearPaint);

                    int heigh = surfaceView.getHeight() / 255;
                    int width = surfaceView.getWidth() / 255;
                    Log.d(TAG, "Surface [High, Width]:" + heigh + ", " + width);

                    for (Detection result : results)
                    {
                        RectF boundingBox = result.getBoundingBox();

                        float top = boundingBox.top * heigh;
                        float bottom = boundingBox.bottom * heigh;
                        float left = boundingBox.left * width;
                        float right = boundingBox.right * width;

                        canvas.drawRect(new RectF(left, top, right, bottom), boxPaint);

                        String drawableText = result.getCategories().get(0).getLabel() + " " +
                                String.format("%.2f", result.getCategories().get(0).getScore());

                        textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length(), bounds);

                        canvas.drawRect(
                                left,
                                top,
                                left + bounds.width() + BOUNDING_RECT_TEXT_PADDING,
                                top + bounds.height() + BOUNDING_RECT_TEXT_PADDING,
                                textBackgroundPaint
                        );

                        canvas.drawText(drawableText, left, top + bounds.height(), textPaint);

                    }
                    sfhTrackHolder.unlockCanvasAndPost(canvas);
                } catch (Exception e) {
                    Log.e(TAG, "Surface close");
                }


                imageProxy.close();
                Log.d(TAG, "ImageProxy close");
            }
        });

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, imageAnalysis, preview);

    }

    private void initPaints() {
        textBackgroundPaint = new Paint();
        textPaint = new Paint();
        boxPaint = new Paint();

        textBackgroundPaint.setColor(Color.BLACK);
        textBackgroundPaint.setStyle(Paint.Style.FILL);
        textBackgroundPaint.setTextSize(50f);

        textPaint.setColor(Color.WHITE);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(50f);

        boxPaint.setColor(ContextCompat.getColor(getContext(), R.color.bounding_box_color));
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8f);

    }

    public void initialObjectDetector() {

        ObjectDetector.ObjectDetectorOptions optionsBuilder =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setScoreThreshold(0.5f)
                        .setMaxResults(3).build();

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(getContext(), MODEL_FILE, optionsBuilder);
                Log.d(TAG, "ObjectDetector ready");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void detectByObjectDetector(TensorImage tensorImage) {
        results  = null;
        if (objectDetector != null)
        {
            results = objectDetector.detect(tensorImage);
        }

        for(Detection result : results)
        {
            Log.d(TAG,"Inference result: " + result.getCategories());

        }
    }

}