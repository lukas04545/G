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
import android.util.Log
import android.view.*
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*

class OverlayService : Service() {

    companion object {
        private const val TAG = "OverlayService"
        const val ACTION_START      = "com.yolodetect.START"
        const val ACTION_STOP       = "com.yolodetect.STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val EXTRA_CONFIDENCE  = "confidence"
        const val EXTRA_MODEL       = "model"
        const val EXTRA_PERSON_ONLY = "person_only"
        private const val NOTIF_ID   = 1
        private const val CHANNEL_ID = "yolo_overlay"
    }

    private val mainHandler  = Handler(Looper.getMainLooper())
    private val scope        = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay:  VirtualDisplay?  = null
    private var imageReader:     ImageReader?      = null
    private var overlayView:     OverlayView?      = null
    private var detector:        YoloDetector?     = null

    private lateinit var windowManager: WindowManager
    private var screenW = 0
    private var screenH = 0

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            Log.i(TAG, "MediaProjection stopped by system")
            stopSelf()
        }
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        // Use current window metrics on API 30+; fall back to getRealMetrics
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager.currentWindowMetrics.bounds
            screenW = bounds.width()
            screenH = bounds.height()
        } else {
            val m = android.util.DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(m)
            screenW = m.widthPixels
            screenH = m.heightPixels
        }
        Log.i(TAG, "Screen: ${screenW}x${screenH}")
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { stopSelf(); return START_NOT_STICKY }
        if (intent?.action != ACTION_START) return START_NOT_STICKY

        // startForeground MUST come before any projection work on Android 10+
        startForeground(NOTIF_ID, buildNotification())

        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
        val resultData: Intent? =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
                intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
            else
                @Suppress("DEPRECATION") intent.getParcelableExtra(EXTRA_RESULT_DATA)
        val confidence  = intent.getFloatExtra(EXTRA_CONFIDENCE, 0.5f)
        val modelFile   = intent.getStringExtra(EXTRA_MODEL) ?: "yolov8n.onnx"
        val personOnly  = intent.getBooleanExtra(EXTRA_PERSON_ONLY, false)

        if (resultCode != Activity.RESULT_OK || resultData == null) {
            Log.e(TAG, "Bad projection result ($resultCode)")
            stopSelf(); return START_NOT_STICKY
        }

        startCapture(resultCode, resultData, confidence, modelFile, personOnly)
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        scope.cancel()
        mainHandler.post { removeOverlay() }
        virtualDisplay?.release()
        imageReader?.close()
        runCatching { mediaProjection?.unregisterCallback(projectionCallback) }
        mediaProjection?.stop()
        detector?.close()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // -------------------------------------------------------------------------
    // Capture setup
    // -------------------------------------------------------------------------

    private fun startCapture(resultCode: Int, data: Intent, confidence: Float,
                             modelFile: String, personOnly: Boolean) {
        val projMgr = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = runCatching {
            projMgr.getMediaProjection(resultCode, data).also {
                it.registerCallback(projectionCallback, mainHandler)
            }
        }.onFailure { Log.e(TAG, "getMediaProjection failed", it); stopSelf() }.getOrNull()
            ?: return

        val dpi = resources.displayMetrics.densityDpi
        imageReader = ImageReader.newInstance(screenW, screenH, PixelFormat.RGBA_8888, 2)

        virtualDisplay = runCatching {
            mediaProjection!!.createVirtualDisplay(
                "YoloCapture", screenW, screenH, dpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader!!.surface, null, mainHandler,
            )
        }.onFailure { Log.e(TAG, "createVirtualDisplay failed", it); stopSelf() }.getOrNull()
            ?: return

        // Add overlay on main thread (we're already there via onStartCommand)
        runCatching { addOverlay() }.onFailure { Log.e(TAG, "addOverlay failed", it); stopSelf(); return }

        launchDetectionLoop(confidence, modelFile, personOnly)
    }

    // -------------------------------------------------------------------------
    // Overlay window
    // -------------------------------------------------------------------------

    private fun addOverlay() {
        val view = OverlayView(this)
        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                    or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                    or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                    or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        )
        windowManager.addView(view, params)
        overlayView = view
        Log.i(TAG, "Overlay added ${screenW}x${screenH}")
    }

    private fun removeOverlay() {
        overlayView?.let { runCatching { windowManager.removeView(it) } }
        overlayView = null
    }

    // -------------------------------------------------------------------------
    // Detection loop
    // -------------------------------------------------------------------------

    private fun launchDetectionLoop(confidence: Float, modelFile: String, personOnly: Boolean) {
        scope.launch {
            runCatching {
                detector = YoloDetector(applicationContext,
                    modelFileName       = modelFile,
                    confidenceThreshold = confidence,
                    personOnly          = personOnly)
                detector!!.load()
                Log.i(TAG, "Model: $modelFile  inputSize=${detector!!.inputSize}  personOnly=$personOnly")
                Log.i(TAG, "Detector ready")

                while (isActive) {
                    val image = try { imageReader?.acquireLatestImage() }
                                catch (_: Exception) { null }
                    if (image == null) { delay(16); continue }

                    runCatching {
                        val plane  = image.planes[0]
                        val stride = plane.rowStride / plane.pixelStride   // pixels per row

                        // Create bitmap from buffer (stride may be wider than screenW)
                        val raw = Bitmap.createBitmap(stride, screenH, Bitmap.Config.ARGB_8888)
                        raw.copyPixelsFromBuffer(plane.buffer)

                        val frame = if (stride != screenW)
                            Bitmap.createBitmap(raw, 0, 0, screenW, screenH).also { raw.recycle() }
                        else raw

                        val dets = detector!!.detect(frame)
                        frame.recycle()
                        overlayView?.update(dets, screenW, screenH)
                    }.onFailure { Log.w(TAG, "frame error: ${it.message}") }

                    image.close()
                    delay(33)
                }
            }.onFailure {
                Log.e(TAG, "Detection loop fatal", it)
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
            .setContentText("Running — pull down to stop")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .addAction(android.R.drawable.ic_delete, "Stop", stopPi)
            .setOngoing(true)
            .build()
    }
}
