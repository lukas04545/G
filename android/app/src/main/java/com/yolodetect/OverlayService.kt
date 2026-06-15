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
import android.util.Log
import android.view.*
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*

class OverlayService : Service() {

    companion object {
        private const val TAG = "OverlayService"
        const val ACTION_START       = "com.yolodetect.START"
        const val ACTION_STOP        = "com.yolodetect.STOP"
        const val EXTRA_RESULT_CODE  = "result_code"
        const val EXTRA_RESULT_DATA  = "result_data"
        const val EXTRA_CONFIDENCE   = "confidence"
        private const val NOTIF_ID   = 1
        private const val CHANNEL_ID = "yolo_overlay"
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var overlayView: OverlayView? = null
    private var detector: YoloDetector? = null

    private lateinit var windowManager: WindowManager
    private var screenW = 0
    private var screenH = 0

    // Stop the projection cleanly if Android revokes it (e.g. on some OEM ROMs)
    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            Log.i(TAG, "MediaProjection stopped externally")
            stopSelf()
        }
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealMetrics(metrics)
        screenW = metrics.widthPixels
        screenH = metrics.heightPixels
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { stopSelf(); return START_NOT_STICKY }
        if (intent?.action != ACTION_START) return START_NOT_STICKY

        // Must call startForeground before doing any projection work on Android 10+
        startForeground(NOTIF_ID, buildNotification())

        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
        val resultData: Intent? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
            intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        else
            @Suppress("DEPRECATION") intent.getParcelableExtra(EXTRA_RESULT_DATA)
        val confidence = intent.getFloatExtra(EXTRA_CONFIDENCE, 0.5f)

        if (resultCode == Activity.RESULT_CANCELED || resultData == null) {
            Log.e(TAG, "Invalid projection result — stopping")
            stopSelf()
            return START_NOT_STICKY
        }

        startCapture(resultCode, resultData, confidence)
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        scope.cancel()
        mainHandler.post { removeOverlay() }
        virtualDisplay?.release()
        imageReader?.close()
        try { mediaProjection?.unregisterCallback(projectionCallback) } catch (_: Exception) {}
        mediaProjection?.stop()
        detector?.close()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // -------------------------------------------------------------------------
    // Capture setup  (runs on main thread via onStartCommand)
    // -------------------------------------------------------------------------

    private fun startCapture(resultCode: Int, data: Intent, confidence: Float) {
        val projMgr = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        try {
            mediaProjection = projMgr.getMediaProjection(resultCode, data).also {
                it.registerCallback(projectionCallback, mainHandler)
            }
        } catch (e: Exception) {
            Log.e(TAG, "getMediaProjection failed", e)
            stopSelf(); return
        }

        imageReader = ImageReader.newInstance(screenW, screenH, PixelFormat.RGBA_8888, 2)

        try {
            virtualDisplay = mediaProjection!!.createVirtualDisplay(
                "YoloCapture", screenW, screenH,
                resources.displayMetrics.densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader!!.surface, null, mainHandler,
            )
        } catch (e: Exception) {
            Log.e(TAG, "createVirtualDisplay failed", e)
            stopSelf(); return
        }

        addOverlay()
        launchDetectionLoop(confidence)
    }

    // -------------------------------------------------------------------------
    // Overlay window  (must be called on main thread)
    // -------------------------------------------------------------------------

    private fun addOverlay() {
        val view = OverlayView(this)
        val params = WindowManager.LayoutParams(
            screenW, screenH,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                    or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                    or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                    or WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            PixelFormat.TRANSLUCENT,
        ).apply { gravity = Gravity.TOP or Gravity.START }
        try {
            windowManager.addView(view, params)
            overlayView = view
        } catch (e: Exception) {
            Log.e(TAG, "addView failed", e)
            stopSelf()
        }
    }

    private fun removeOverlay() {
        overlayView?.let { v ->
            try { windowManager.removeView(v) } catch (_: Exception) {}
        }
        overlayView = null
    }

    // -------------------------------------------------------------------------
    // Detection loop  (runs on Dispatchers.Default)
    // -------------------------------------------------------------------------

    private fun launchDetectionLoop(confidence: Float) {
        scope.launch {
            try {
                detector = YoloDetector(applicationContext, confidenceThreshold = confidence)
                detector!!.load()
                Log.i(TAG, "Detector ready, starting loop")

                while (isActive) {
                    val image = try {
                        imageReader?.acquireLatestImage()
                    } catch (e: Exception) {
                        Log.w(TAG, "acquireLatestImage: ${e.message}")
                        null
                    }

                    if (image == null) { delay(16); continue }

                    try {
                        val plane = image.planes[0]
                        val rowStridePx = plane.rowStride / plane.pixelStride

                        val raw = Bitmap.createBitmap(rowStridePx, screenH, Bitmap.Config.ARGB_8888)
                        raw.copyPixelsFromBuffer(plane.buffer)

                        // Trim padding columns if rowStride > actual width
                        val frame = if (rowStridePx != screenW)
                            Bitmap.createBitmap(raw, 0, 0, screenW, screenH)
                        else raw

                        val dets = detector!!.detect(frame)
                        overlayView?.update(dets, screenW, screenH)

                        if (rowStridePx != screenW) raw.recycle()
                        frame.recycle()
                    } catch (e: Exception) {
                        Log.w(TAG, "Frame processing error: ${e.message}")
                    } finally {
                        image.close()
                    }
                    delay(33)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Detection loop fatal error", e)
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
        val stopIntent = PendingIntent.getService(
            this, 0,
            Intent(this, OverlayService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("YOLO Contrast Enhance — running")
            .setContentText("Tap Stop to disable the overlay")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .addAction(android.R.drawable.ic_delete, "Stop", stopIntent)
            .setOngoing(true)
            .build()
    }
}
