package com.yolodetect

import android.content.Context
import android.graphics.*
import android.view.View

/**
 * Full-screen transparent overlay drawn via WindowManager on top of all apps.
 * Renders a soft contrast-boost glow border around each detected person,
 * matching the visual effect of hardware "Shadow Boost" / enemy-highlight
 * modes found on high-end gaming monitors.
 */
class OverlayView(context: Context) : View(context) {

    private var detections: List<Detection> = emptyList()
    private var frameW = 1f
    private var frameH = 1f

    // Outer glow — wide, semi-transparent
    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 18f
        maskFilter = BlurMaskFilter(24f, BlurMaskFilter.Blur.OUTER)
    }

    // Inner crisp border
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    // Label background + text
    private val tagBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val tagTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.MONOSPACE
    }

    fun update(dets: List<Detection>, srcW: Int, srcH: Int) {
        detections = dets
        frameW = srcW.toFloat()
        frameH = srcH.toFloat()
        postInvalidate()
    }

    fun clear() { detections = emptyList(); postInvalidate() }

    override fun onDraw(canvas: Canvas) {
        if (detections.isEmpty()) return
        val vw = width.toFloat()
        val vh = height.toFloat()

        for (det in detections) {
            val color = glowColor(det.classId)

            val l = det.bbox.left   / frameW * vw
            val t = det.bbox.top    / frameH * vh
            val r = det.bbox.right  / frameW * vw
            val b = det.bbox.bottom / frameH * vh
            val rect = RectF(l, t, r, b)

            // Outer glow
            glowPaint.color = Color.argb(120, Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRoundRect(rect, 8f, 8f, glowPaint)

            // Crisp inner border
            borderPaint.color = color
            canvas.drawRoundRect(rect, 8f, 8f, borderPaint)

            // Confidence tag
            val label = "${"%.0f".format(det.confidence * 100)}%"
            val tw = tagTextPaint.measureText(label)
            val th = tagTextPaint.textSize
            val tagL = l
            val tagT = if (t > th + 8) t - th - 8 else b

            tagBgPaint.color = Color.argb(180, Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRoundRect(tagL, tagT, tagL + tw + 12, tagT + th + 6, 4f, 4f, tagBgPaint)
            canvas.drawText(label, tagL + 6, tagT + th, tagTextPaint)
        }
    }

    // Warm highlight palette: amber → orange → red, cycling per class
    private fun glowColor(classId: Int): Int {
        val hue = (30f + classId * 40f) % 60f + 10f   // stays in warm amber-orange band
        return Color.HSVToColor(floatArrayOf(hue, 1f, 1f))
    }
}
