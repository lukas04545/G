package com.yolodetect

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.yolodetect.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "YoloDetect"
        private const val REQ_CAMERA   = 10
        private const val REQ_OVERLAY  = 11
        private const val REQ_CAPTURE  = 12
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private val inferenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var detector: YoloDetector? = null
    private var frameWidth  = YoloDetector.INPUT_SIZE
    private var frameHeight = YoloDetector.INPUT_SIZE
    private var lastFrameMs = System.currentTimeMillis()
    private var frameCount  = 0
    private var avgLatencyMs = 0.0

    // Current mode
    private var screenMode = false

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Mode toggle button
        binding.btnToggleMode.setOnClickListener { toggleMode() }
        // Confidence slider wires into the detector live
        binding.confidenceSlider.addOnChangeListener { _, value, _ ->
            detector?.let {
                // rebuild detector with new threshold for camera mode
                if (!screenMode) reloadDetector(value / 100f)
            }
            binding.confidenceLabel.text = "Confidence: ${value.toInt()}%"
        }

        loadDetector(binding.confidenceSlider.value / 100f)
        startCameraMode()  // start in camera mode by default
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
        cameraExecutor.shutdown()
        detector?.close()
        // Make sure overlay service is stopped when activity is closed
        stopService(Intent(this, OverlayService::class.java))
    }

    // -------------------------------------------------------------------------
    // Mode switching
    // -------------------------------------------------------------------------

    private fun toggleMode() {
        if (!screenMode) switchToScreenMode() else switchToCameraMode()
    }

    private fun switchToScreenMode() {
        // 1. Need SYSTEM_ALERT_WINDOW
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Grant 'Display over other apps' permission first", Toast.LENGTH_LONG).show()
            startActivityForResult(
                Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")),
                REQ_OVERLAY,
            )
            return
        }
        // 2. Request MediaProjection
        val mgr = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(mgr.createScreenCaptureIntent(), REQ_CAPTURE)
    }

    private fun switchToCameraMode() {
        screenMode = false
        binding.btnToggleMode.text = "Switch to Screen Mode"
        binding.previewView.visibility = android.view.View.VISIBLE
        binding.boundingBoxView.visibility = android.view.View.VISIBLE
        stopService(Intent(this, OverlayService::class.java))
        startCameraMode()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            REQ_OVERLAY -> {
                if (Settings.canDrawOverlays(this)) switchToScreenMode()
                else Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
            }
            REQ_CAPTURE -> {
                if (resultCode == Activity.RESULT_OK && data != null) {
                    screenMode = true
                    binding.btnToggleMode.text = "Switch to Camera Mode"
                    binding.previewView.visibility = android.view.View.GONE
                    binding.boundingBoxView.visibility = android.view.View.GONE
                    binding.statsText.text = "Screen mode active — switch apps to play"

                    val confidence = binding.confidenceSlider.value / 100f
                    startService(
                        Intent(this, OverlayService::class.java)
                            .setAction(OverlayService.ACTION_START)
                            .putExtra(OverlayService.EXTRA_RESULT_CODE, resultCode)
                            .putExtra(OverlayService.EXTRA_RESULT_DATA, data)
                            .putExtra(OverlayService.EXTRA_CONFIDENCE, confidence)
                    )
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Camera mode
    // -------------------------------------------------------------------------

    private fun startCameraMode() {
        if (hasCameraPermission()) bindCamera() else requestCameraPermission()
    }

    private fun bindCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            androidx.camera.core.resolutionselector.ResolutionStrategy(
                                Size(640, 640),
                                androidx.camera.core.resolutionselector.ResolutionStrategy
                                    .FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            )
                        ).build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(proxy: ImageProxy) {
        if (screenMode || detector == null) { proxy.close(); return }
        val bitmap: Bitmap
        try {
            bitmap = Bitmap.createBitmap(proxy.width, proxy.height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(proxy.planes[0].buffer)
        } finally { proxy.close() }

        frameWidth  = bitmap.width
        frameHeight = bitmap.height

        inferenceScope.launch {
            val t0 = System.currentTimeMillis()
            val dets = runCatching { detector!!.detect(bitmap) }.getOrDefault(emptyList())
            val elapsed = System.currentTimeMillis() - t0
            frameCount++
            avgLatencyMs += (elapsed - avgLatencyMs) / frameCount
            val fps = 1000.0 / (System.currentTimeMillis() - lastFrameMs).coerceAtLeast(1)
            lastFrameMs = System.currentTimeMillis()

            withContext(Dispatchers.Main) {
                binding.boundingBoxView.setDetections(dets, frameWidth, frameHeight)
                binding.statsText.text =
                    "FPS: %.1f  |  Latency: %dms  |  Objects: %d".format(fps, elapsed, dets.size)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Detector
    // -------------------------------------------------------------------------

    private fun loadDetector(confidence: Float) {
        inferenceScope.launch {
            try {
                val d = YoloDetector(applicationContext, confidenceThreshold = confidence)
                d.load()
                detector = d
                withContext(Dispatchers.Main) {
                    binding.statsText.text = "Model ready"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                withContext(Dispatchers.Main) {
                    binding.statsText.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun reloadDetector(confidence: Float) {
        detector?.close()
        detector = null
        loadDetector(confidence)
    }

    // -------------------------------------------------------------------------
    // Permissions
    // -------------------------------------------------------------------------

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() =
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == REQ_CAMERA && results.firstOrNull() == PackageManager.PERMISSION_GRANTED)
            bindCamera()
        else
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }
}
