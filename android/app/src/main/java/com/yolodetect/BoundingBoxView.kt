package com.yolodetect

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class BoundingBoxView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    init {
        // Software layer required for BlurMaskFilter on mask glow
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    // Mask: outer glow (blurred, expanded)
    private val maskGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        maskFilter = BlurMaskFilter(18f, BlurMaskFilter.Blur.OUTER)
    }

    // Mask: filled silhouette
    private val maskFillPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    // Fallback: bounding box stroke
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        typeface = Typeface.DEFAULT_BOLD
    }

    private val tagBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private var detections: List<Detection> = emptyList()
    private var scaleX: Float = 1f
    private var scaleY: Float = 1f

    fun setDetections(dets: List<Detection>, frameWidth: Int, frameHeight: Int) {
        scaleX = frameWidth.toFloat()
        scaleY = frameHeight.toFloat()
        detections = dets
        postInvalidate()
    }

    fun clear() {
        detections = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        val vw = width.toFloat()
        val vh = height.toFloat()

        for (det in detections) {
            val color = colorForClass(det.classId)
            val l = det.bbox.left   / scaleX * vw
            val t = det.bbox.top    / scaleY * vh
            val r = det.bbox.right  / scaleX * vw
            val b = det.bbox.bottom / scaleY * vh
            val dst = RectF(l, t, r, b)

            if (det.mask != null) {
                // Glow: draw mask scaled to expanded rect
                val exp = 12f
                maskGlowPaint.color = Color.argb(160,
                    Color.red(color), Color.green(color), Color.blue(color))
                canvas.drawBitmap(det.mask, null,
                    RectF(l - exp, t - exp, r + exp, b + exp), maskGlowPaint)

                // Filled silhouette at partial alpha
                maskFillPaint.color = Color.argb(110,
                    Color.red(color), Color.green(color), Color.blue(color))
                canvas.drawBitmap(det.mask, null, dst, maskFillPaint)
            } else {
                boxPaint.color = color
                canvas.drawRect(dst, boxPaint)
            }

            // Label
            val label = "${det.className} ${"%.0f".format(det.confidence * 100)}%"
            val tw = textPaint.measureText(label)
            val th = textPaint.textSize
            val tagT = if (t > th + 8) t - th - 8 else b + 4
            tagBgPaint.color = Color.argb(200,
                Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRoundRect(l, tagT, l + tw + 12, tagT + th + 6, 5f, 5f, tagBgPaint)
            canvas.drawText(label, l + 6, tagT + th, textPaint)
        }
    }

    private fun colorForClass(classId: Int): Int {
        val hue = (classId * 137.508f) % 360f
        return Color.HSVToColor(floatArrayOf(hue, 0.85f, 0.95f))
    }
}
