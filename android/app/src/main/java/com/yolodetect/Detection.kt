package com.yolodetect

import android.graphics.RectF

/** A single object detected in one frame. */
data class Detection(
    /** Bounding box in the *original camera frame* coordinate space. */
    val bbox: RectF,
    val confidence: Float,
    val classId: Int,
    val className: String,
)
