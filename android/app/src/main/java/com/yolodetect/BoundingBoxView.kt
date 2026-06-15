package com.yolodetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Transparent overlay that draws bounding boxes and labels on top of the camera preview.
 *
 * Bounding boxes are supplied in *normalised* coordinates [0, 1] so the view
 * doesn't need to know the camera resolution.
 */
class BoundingBoxView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        textSize = 36f
    }

    // Normalised detections: bbox in [0, 1] range
    private var detections: List<Detection> = emptyList()
    private var scaleX: Float = 1f
    private var scaleY: Float = 1f

    /**
     * Update displayed detections.
     *
     * @param dets Detections with bbox in *original camera frame* pixel coords.
     * @param frameWidth Width of the camera frame used for detection.
     * @param frameHeight Height of the camera frame used for detection.
     */
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
            boxPaint.color = color

            // Map from camera frame coords → view coords
            val left   = det.bbox.left   / scaleX * vw
            val top    = det.bbox.top    / scaleY * vh
            val right  = det.bbox.right  / scaleX * vw
            val bottom = det.bbox.bottom / scaleY * vh

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = "${det.className} ${"%.0f".format(det.confidence * 100)}%"
            val textW  = textPaint.measureText(label)
            val textH  = textPaint.textSize
            val labelY = if (top > textH + 8) top else top + textH + 8

            bgPaint.color = color
            canvas.drawRect(left, labelY - textH - 4, left + textW + 8, labelY + 4, bgPaint)
            canvas.drawText(label, left + 4, labelY, textPaint)
        }
    }

    // Deterministic per-class colour (evenly spaced around HSV hue wheel)
    private fun colorForClass(classId: Int): Int {
        val hue = (classId * 137.508f) % 360f   // golden-angle spacing
        return Color.HSVToColor(floatArrayOf(hue, 0.85f, 0.95f))
    }
}
