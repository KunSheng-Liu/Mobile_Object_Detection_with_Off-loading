package com.example.final_project_ver1.ui.notifications;

import static android.content.ContentValues.TAG;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.LifecycleOwner;
import androidx.lifecycle.ViewModelProvider;

import com.example.final_project_ver1.R;
import com.example.final_project_ver1.databinding.FragmentDashboardBinding;
import com.example.final_project_ver1.databinding.FragmentNotificationsBinding;
import com.google.android.gms.common.util.IOUtils;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class NotificationsFragment extends Fragment {

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
    private FragmentNotificationsBinding binding;

    private int[] tf_InputDim;
    private int[] tf_OutputDim;
    private static final String MODEL_FILE = "mobilenetv1.tflite";

    ObjectDetector objectDetector;

    public static Socket s;
    private static String sockectHost = "192.168.0.106";
    private static int sockectPort = 9851;

    public static boolean issocketconnected = false;

    class Result {
        private RectF rectF;
        private float accuracy;
        private int labelID;
        private String category;

        public Result (RectF _rectF, float _accuracy, int _label, String _category) {
            rectF = _rectF;
            accuracy = _accuracy;
            labelID = _label;
            category = _category;
        }

        public float getAccuracy() {
            return accuracy;
        }

        public int getLabelID() {
            return labelID;
        }

        public RectF getRectF() {
            return rectF;
        }

        public String getCategory() {
            return category;
        }
    }

    ArrayList<Result> results = new ArrayList<>();
    ArrayList<String> labels = new ArrayList<>();

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        binding = FragmentNotificationsBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        /**
         * Socket Connect
         */
        createSocket createsocket = new createSocket();
        createsocket.execute();

        initPaints();

        try {
            AssetManager am = getContext().getAssets();
            InputStream inputStream = am.open("yolov4labels.txt");
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));

            String label;
            while ((label = bufferedReader.readLine()) != null) {
                labels.add(label);


            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        previewView = binding.prevNotification;
        surfaceView = binding.sfvNotification;
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

        disconnectSocket senddisconnect = new disconnectSocket();
        senddisconnect.execute();
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
                 * Image Compress (Quality 1~100)
                 * */

                Matrix matrix = new Matrix();
                matrix.postRotate(90);

                Bitmap rotatedBitmap = Bitmap.createBitmap(bitmapCamera, 0, 0, bitmapCamera.getWidth(), bitmapCamera.getHeight(), matrix, true);

                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, baos); //compressQ


                String serverMessage = "";
                if(issocketconnected)
                {
                    byte[] imagebytes = baos.toByteArray();

                    int transmitted_bytesize = imagebytes.length;
                    Log.d(TAG, "Image transmmit size: " + transmitted_bytesize);

                    try {
//                        InputStreamReader inputStreamReader = null;
//                        inputStreamReader = new InputStreamReader(s.getInputStream());
//                        BufferedReader in = new BufferedReader( inputStreamReader );

                        OutputStream out = s.getOutputStream();
                        DataOutputStream dataOutputStream = new DataOutputStream(out);
                        dataOutputStream.write(imagebytes, 0, imagebytes.length);
                        Log.d(TAG, "Finish sending");

//                        serverMessage = in.readLine();
                        Log.d(TAG, "Finish reveiving");

                        out.flush();
                        dataOutputStream.flush();

                    } catch (IOException e) {
                        Log.e(TAG, "Socket error");
                    }
                }

//                if (serverMessage != null) {
//                    String[] receives = serverMessage.split(" ");
//
//                    int heigh = surfaceView.getHeight();
//                    int width = surfaceView.getWidth();
//                    Log.d(TAG, "Surface [High, Width]:" + heigh + ", " + width);
//
//                    for (int i = 0; i < receives.length / 6; i++) {
//                        float top = Float.valueOf(receives[i * 6]) * heigh;
//                        float bottom = Float.valueOf(receives[i * 6 + 1]) * heigh;
//                        float left = Float.valueOf(receives[i * 6 + 2]) * width;
//                        float right = Float.valueOf(receives[i * 6 + 3]) * width;
//                        float accuracy = Float.valueOf(receives[i * 6 + 4]);
//                        int labelID = Math.round(Float.valueOf(receives[i * 6 + 5]));
//
//                        results.add(new Result(new RectF(left, top, right, bottom), accuracy, labelID, labels.get(labelID)));
//                    }

                    /**
                     * Start detect
                     */
//                    Canvas canvas = null;
//                    try {
//                        canvas = sfhTrackHolder.lockCanvas();
////                    Paint clearPaint = new Paint();
////                    clearPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
////                    canvas.drawRect(0, 0, canvas.getWidth(), canvas.getHeight(), clearPaint);
//
//
//                        for (Result result : results) {
//                            RectF boundingBox = result.getRectF();
//
//                            float top = boundingBox.top * heigh;
//                            float bottom = boundingBox.bottom * heigh;
//                            float left = boundingBox.left * width;
//                            float right = boundingBox.right * width;
//
//                            canvas.drawRect(new RectF(left, top, right, bottom), boxPaint);
//
//                            String drawableText = result.getCategory() + " " +
//                                    String.format("%.2f", result.getAccuracy());
//
//                            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length(), bounds);
//
//                            canvas.drawRect(
//                                    left,
//                                    top,
//                                    left + bounds.width() + BOUNDING_RECT_TEXT_PADDING,
//                                    top + bounds.height() + BOUNDING_RECT_TEXT_PADDING,
//                                    textBackgroundPaint
//                            );
//
//                            canvas.drawText(drawableText, left, top + bounds.height(), textPaint);
//
//                        }
//                        sfhTrackHolder.unlockCanvasAndPost(canvas);
//                    } catch (Exception e) {
//                        Log.e(TAG, "Surface close");
//                    }
//                }

//                issocketconnected = false;
                imageProxy.close();
                Log.d(TAG, "ImageProxy close");
            }
        });

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, imageAnalysis, preview);

    }

    public static class createSocket extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... voids) {
            try {
                Thread.currentThread().setName("CsPEKO");
                s = new Socket(sockectHost, sockectPort); //192.168.0.106 140.118.206.27

                if (s.isConnected()) {
                    Log.w("delayinfo_debug", " Socket is connected :)");
                    issocketconnected = true;
                }

            } catch (IOException e) {
                Log.e(TAG, "Create Socket error!");
                e.printStackTrace();
            }
            return null;
        }
    }

    public static class disconnectSocket extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... voids) {
            try {
                if (issocketconnected) {
                    Thread.currentThread().setName("DsPEKO");
                    Log.w(TAG, "Disconnect socket...");
                    issocketconnected = false;
                    s.close();
                }

            } catch (UnknownHostException e) {
                Log.e(TAG, "Disconnect Socket error! (UnknownHostException)");
                e.printStackTrace();
            } catch (IOException e) {
                Log.e(TAG, "Disconnect Socket error! (IOException)");
                e.printStackTrace();
            }
            return null;
        }

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

}