package com.example.final_project_ver1;

import android.app.Activity;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.task.vision.detector.Detection;

import java.util.List;

public class OverlaySurface extends Activity implements SurfaceHolder.Callback {

    List<Detection> results;
    private Paint boxPaint;
    private Paint textBackgroundPaint;
    private Paint textPaint;
    private Rect bounds = new Rect();
    private int BOUNDING_RECT_TEXT_PADDING = 8;

    private static final String TAG = "Overlay SurfaceView";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        SurfaceView view = new SurfaceView(this);
        setContentView(view);
        view.getHolder().addCallback(this);

        initPaints();
    }

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder) {
        tryDrawing(holder);
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int i, int i1, int i2) {
        tryDrawing(holder);
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder) {

    }

    private void initPaints() {
        textBackgroundPaint.setColor(Color.BLACK);
        textBackgroundPaint.setStyle(Paint.Style.FILL);
        textBackgroundPaint.setTextSize(50f);

        textPaint.setColor(Color.BLACK);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(50f);

        boxPaint.setColor(ContextCompat.getColor(this, R.color.bounding_box_color));
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8f);

    }

    private void tryDrawing(SurfaceHolder holder) {
        Log.i(TAG, "Trying to draw...");

        Canvas canvas = holder.lockCanvas();
        if (canvas == null) {
            Log.e(TAG, "Cannot draw onto the canvas as it's null");
        } else {
            draw(canvas);
            holder.unlockCanvasAndPost(canvas);
        }
    }

    public void draw(final Canvas canvas) {

        for (Detection result : results) {
            RectF boundingBox = result.getBoundingBox();

            float top = boundingBox.top;
            float bottom = boundingBox.bottom;
            float left = boundingBox.left;
            float right = boundingBox.right;

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
    }


    public void setResult(List<Detection> detectionResults) {
        results = detectionResults;
    }



}
