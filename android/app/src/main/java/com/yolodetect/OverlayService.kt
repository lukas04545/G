package com.yolodetect

import android.app.*
import android.content.*
import android.graphics.*
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.*
import android.util.DisplayMetrics
import android.view.*
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*

/**
 * Foreground service that:
 * 1. Captures the screen via MediaProjection
 * 2. Runs YoloDetector on each frame
 * 3. Draws a contrast-boost overlay via WindowManager
 */
class OverlayService : Service() {

    companion object {
        const val ACTION_START  = "com.yolodetect.START"
        const val ACTION_STOP   = "com.yolodetect.STOP"
        const val EXTRA_RESULT_CODE   = "result_code"
        const val EXTRA_RESULT_DATA   = "result_data"
        const val EXTRA_CONFIDENCE    = "confidence"
        private const val NOTIF_ID    = 1
        private const val CHANNEL_ID  = "yolo_overlay"
    }

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var overlayView: OverlayView? = null
    private var detector: YoloDetector? = null

    private lateinit var windowManager: WindowManager
    private lateinit var metrics: DisplayMetrics

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealMetrics(metrics)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
                val resultData = intent.getParcelableExtra<Intent>(EXTRA_RESULT_DATA)!!
                val confidence = intent.getFloatExtra(EXTRA_CONFIDENCE, 0.5f)
                startForeground(NOTIF_ID, buildNotification())
                startCapture(resultCode, resultData, confidence)
            }
            ACTION_STOP -> stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        scope.cancel()
        removeOverlay()
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()
        detector?.close()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // -------------------------------------------------------------------------
    // Screen capture setup
    // -------------------------------------------------------------------------

    private fun startCapture(resultCode: Int, data: Intent, confidence: Float) {
        val projMgr = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = projMgr.getMediaProjection(resultCode, data)

        val w = metrics.widthPixels
        val h = metrics.heightPixels
        val dpi = metrics.densityDpi

        imageReader = ImageReader.newInstance(w, h, PixelFormat.RGBA_8888, 2)

        virtualDisplay = mediaProjection!!.createVirtualDisplay(
            "YoloCapture", w, h, dpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader!!.surface, null, null,
        )

        addOverlay(w, h)
        loadAndRun(confidence, w, h)
    }

    // -------------------------------------------------------------------------
    // Overlay window
    // -------------------------------------------------------------------------

    private fun addOverlay(w: Int, h: Int) {
        overlayView = OverlayView(this)
        val params = WindowManager.LayoutParams(
            w, h,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                    or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                    or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
        }
        windowManager.addView(overlayView, params)
    }

    private fun removeOverlay() {
        overlayView?.let {
            try { windowManager.removeView(it) } catch (_: Exception) {}
        }
        overlayView = null
    }

    // -------------------------------------------------------------------------
    // Detection loop
    // -------------------------------------------------------------------------

    private fun loadAndRun(confidence: Float, w: Int, h: Int) {
        scope.launch {
            try {
                detector = YoloDetector(applicationContext, confidenceThreshold = confidence)
                detector!!.load()

                while (isActive) {
                    val image = imageReader?.acquireLatestImage()
                    if (image == null) { delay(16); continue }
                    try {
                        val plane = image.planes[0]
                        val bitmap = Bitmap.createBitmap(
                            plane.rowStride / plane.pixelStride,
                            h, Bitmap.Config.ARGB_8888,
                        )
                        bitmap.copyPixelsFromBuffer(plane.buffer)

                        // Crop to actual screen width (row stride may be wider)
                        val cropped = if (bitmap.width != w)
                            Bitmap.createBitmap(bitmap, 0, 0, w, h) else bitmap

                        val dets = detector!!.detect(cropped)
                        overlayView?.update(dets, w, h)
                    } finally {
                        image.close()
                    }
                    delay(33) // ~30 fps cap
                }
            } catch (e: Exception) {
                // Model missing or other fatal error — stop silently
                stopSelf()
            }
        }
    }

    // -------------------------------------------------------------------------
    // Notification
    // -------------------------------------------------------------------------

    private fun createNotificationChannel() {
        val ch = NotificationChannel(CHANNEL_ID, "YOLO Overlay", NotificationManager.IMPORTANCE_LOW)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(ch)
    }

    private fun buildNotification(): Notification {
        val stopPi = PendingIntent.getService(
            this, 0,
            Intent(this, OverlayService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("YOLO Contrast Enhance")
            .setContentText("Highlighting detected objects on screen")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .addAction(android.R.drawable.ic_delete, "Stop", stopPi)
            .setOngoing(true)
            .build()
    }
}
