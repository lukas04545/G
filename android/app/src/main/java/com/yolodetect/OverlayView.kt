package com.yolodetect

import android.content.Context
import android.graphics.*
import android.view.View
import java.util.concurrent.atomic.AtomicReference

/**
 * Full-screen transparent overlay drawn in software mode (required for BlurMaskFilter).
 * Safe to call update() from any thread.
 */
class OverlayView(context: Context) : View(context) {

    private data class Frame(
        val detections: List<Detection>,
        val srcW: Float,
        val srcH: Float,
    )

    private val frame = AtomicReference(Frame(emptyList(), 1f, 1f))

    init {
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    // Mask: outer glow
    private val maskGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        maskFilter = BlurMaskFilter(30f, BlurMaskFilter.Blur.OUTER)
    }

    // Mask: filled silhouette
    private val maskFillPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    // Fallback box paints (used when no mask is available)
    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 22f
        maskFilter = BlurMaskFilter(28f, BlurMaskFilter.Blur.OUTER)
    }
    private val midPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f
        color = Color.WHITE
    }

    // Label
    private val tagBgPaint   = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val tagTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        typeface = Typeface.DEFAULT_BOLD
    }

    // HUD (always visible so user knows the overlay is active)
    private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 0, 255, 0)
        textSize = 30f
        typeface = Typeface.MONOSPACE
    }

    fun update(dets: List<Detection>, srcW: Int, srcH: Int) {
        frame.set(Frame(dets, srcW.toFloat(), srcH.toFloat()))
        postInvalidate()
    }

    fun clear() {
        frame.set(Frame(emptyList(), 1f, 1f))
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val f  = frame.get()
        val vw = width.toFloat()
        val vh = height.toFloat()

        canvas.drawText("YOLO ✓  ${f.detections.size} obj", 20f, 60f, hudPaint)
        if (f.detections.isEmpty()) return

        val sx = vw / f.srcW
        val sy = vh / f.srcH

        for (det in f.detections) {
            val color = glowColor(det.classId)
            val l = det.bbox.left   * sx
            val t = det.bbox.top    * sy
            val r = det.bbox.right  * sx
            val b = det.bbox.bottom * sy
            val dst = RectF(l, t, r, b)

            if (det.mask != null) {
                // Outer glow: draw mask at expanded rect with blur
                val exp = 20f
                maskGlowPaint.color = Color.argb(200,
                    Color.red(color), Color.green(color), Color.blue(color))
                canvas.drawBitmap(det.mask, null,
                    RectF(l - exp, t - exp, r + exp, b + exp), maskGlowPaint)

                // Filled silhouette
                maskFillPaint.color = Color.argb(120,
                    Color.red(color), Color.green(color), Color.blue(color))
                canvas.drawBitmap(det.mask, null, dst, maskFillPaint)
            } else {
                // Fallback: glowing rounded rect
                glowPaint.color = Color.argb(160,
                    Color.red(color), Color.green(color), Color.blue(color))
                canvas.drawRoundRect(dst, 10f, 10f, glowPaint)
                midPaint.color = color
                canvas.drawRoundRect(dst, 10f, 10f, midPaint)
                canvas.drawRoundRect(dst, 10f, 10f, borderPaint)
            }

            // Label
            val label = "${det.className} ${"%.0f".format(det.confidence * 100)}%"
            val tw = tagTextPaint.measureText(label)
            val th = tagTextPaint.textSize
            val tagT = if (t > th + 10) t - th - 10 else b + 4
            tagBgPaint.color = Color.argb(200,
                Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRoundRect(l, tagT, l + tw + 14, tagT + th + 6, 5f, 5f, tagBgPaint)
            canvas.drawText(label, l + 7, tagT + th, tagTextPaint)
        }
    }

    // Warm amber-orange palette — stands out on most backgrounds
    private fun glowColor(classId: Int) =
        Color.HSVToColor(floatArrayOf((20f + classId * 47f) % 60f + 5f, 1f, 1f))
}
