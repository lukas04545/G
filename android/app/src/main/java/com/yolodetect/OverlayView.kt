package com.yolodetect

import android.content.Context
import android.graphics.*
import android.os.Build
import android.view.View
import java.util.concurrent.atomic.AtomicReference

/**
 * Full-screen transparent overlay drawn via WindowManager.
 * Thread-safe: update() can be called from any thread.
 */
class OverlayView(context: Context) : View(context) {

    private data class Frame(val detections: List<Detection>, val w: Float, val h: Float)

    // AtomicReference makes the swap visible to onDraw on the main thread
    private val frame = AtomicReference(Frame(emptyList(), 1f, 1f))

    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 18f
        maskFilter = BlurMaskFilter(24f, BlurMaskFilter.Blur.OUTER)
    }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    private val tagBgPaint  = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val tagTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.MONOSPACE
    }

    /** Safe to call from any thread. */
    fun update(dets: List<Detection>, srcW: Int, srcH: Int) {
        frame.set(Frame(dets, srcW.toFloat(), srcH.toFloat()))
        postInvalidate()
    }

    fun clear() { frame.set(Frame(emptyList(), 1f, 1f)); postInvalidate() }

    override fun onDraw(canvas: Canvas) {
        val f = frame.get()
        if (f.detections.isEmpty()) return

        val vw = width.toFloat()
        val vh = height.toFloat()

        for (det in f.detections) {
            val color = glowColor(det.classId)
            val l = det.bbox.left   / f.w * vw
            val t = det.bbox.top    / f.h * vh
            val r = det.bbox.right  / f.w * vw
            val b = det.bbox.bottom / f.h * vh
            val rect = RectF(l, t, r, b)

            glowPaint.color = Color.argb(130, Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRoundRect(rect, 8f, 8f, glowPaint)

            borderPaint.color = color
            canvas.drawRoundRect(rect, 8f, 8f, borderPaint)

            val label = "${"%.0f".format(det.confidence * 100)}%"
            val tw = tagTextPaint.measureText(label)
            val th = tagTextPaint.textSize
            val tagT = if (t > th + 8) t - th - 8 else b
            tagBgPaint.color = Color.argb(190, Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRoundRect(l, tagT, l + tw + 12, tagT + th + 6, 4f, 4f, tagBgPaint)
            canvas.drawText(label, l + 6, tagT + th, tagTextPaint)
        }
    }

    private fun glowColor(classId: Int) =
        Color.HSVToColor(floatArrayOf((30f + classId * 40f) % 60f + 10f, 1f, 1f))
}
