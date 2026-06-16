package com.yolodetect

import android.graphics.Bitmap
import android.graphics.RectF

data class Detection(
    val bbox: RectF,
    val confidence: Float,
    val classId: Int,
    val className: String,
    val mask: Bitmap? = null,
)
