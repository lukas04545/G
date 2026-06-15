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
        private const val TAG        = "YoloDetect"
        private const val REQ_CAMERA  = 10
        private const val REQ_OVERLAY = 11
        private const val REQ_CAPTURE = 12

        private const val MODEL_QUALITY = "yolov8n.onnx"
        private const val MODEL_FAST    = "yolov8n_fast.onnx"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private val inferenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var detector: YoloDetector? = null
    private var screenMode = false

    // Current settings (kept in sync with UI)
    private var useQualityModel = true   // true = 640, false = 320
    private var personOnly      = false
    private var debugMode       = false
    private var confidence      = 0.20f

    private var lastFrameMs = System.currentTimeMillis()
    private var frameCount  = 0
    private var avgLatencyMs = 0.0

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraExecutor = Executors.newSingleThreadExecutor()

        wireControls()
        loadDetector()
        startCameraMode()
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
        cameraExecutor.shutdown()
        detector?.close()
        stopService(Intent(this, OverlayService::class.java))
    }

    // -------------------------------------------------------------------------
    // UI wiring
    // -------------------------------------------------------------------------

    private fun wireControls() {
        // Quality / Fast toggle
        binding.modelQualityGroup.setOnCheckedChangeListener { _, id ->
            useQualityModel = (id == R.id.rbQuality)
            onSettingsChanged()
        }

        // Person-only switch
        binding.switchPersonOnly.setOnCheckedChangeListener { _, checked ->
            personOnly = checked
            onSettingsChanged()
        }

        // Debug switch
        binding.switchDebug.setOnCheckedChangeListener { _, checked ->
            debugMode = checked
            // Debug ignores personOnly and confidence — reflect that in label
            binding.confidenceLabel.text =
                if (checked) "Confidence: ${(confidence * 100).toInt()}%  [DEBUG — showing top 10 raw]"
                else "Confidence: ${(confidence * 100).toInt()}%"
        }

        // Confidence slider
        binding.confidenceSlider.addOnChangeListener { _, value, _ ->
            confidence = value / 100f
            binding.confidenceLabel.text = "Confidence: ${value.toInt()}%"
            onSettingsChanged()
        }

        // Camera ↔ Screen mode button
        binding.btnToggleMode.setOnClickListener {
            if (!screenMode) switchToScreenMode() else switchToCameraMode()
        }
    }

    private fun onSettingsChanged() {
        if (screenMode) {
            // Restart the overlay service with new settings
            stopService(Intent(this, OverlayService::class.java))
            // Re-request screen capture permission flow
            val mgr = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            startActivityForResult(mgr.createScreenCaptureIntent(), REQ_CAPTURE)
        } else {
            // Reload detector for camera mode
            loadDetector()
        }
    }

    // -------------------------------------------------------------------------
    // Mode switching
    // -------------------------------------------------------------------------

    private fun switchToScreenMode() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Grant 'Display over other apps' first", Toast.LENGTH_LONG).show()
            startActivityForResult(
                Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")),
                REQ_OVERLAY,
            )
            return
        }
        val mgr = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(mgr.createScreenCaptureIntent(), REQ_CAPTURE)
    }

    private fun switchToCameraMode() {
        screenMode = false
        binding.btnToggleMode.text = "Switch to Screen Mode"
        binding.previewView.visibility  = android.view.View.VISIBLE
        binding.boundingBoxView.visibility = android.view.View.VISIBLE
        stopService(Intent(this, OverlayService::class.java))
        loadDetector()
        bindCamera()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            REQ_OVERLAY -> if (Settings.canDrawOverlays(this)) switchToScreenMode()
            REQ_CAPTURE -> {
                if (resultCode == Activity.RESULT_OK && data != null) {
                    screenMode = true
                    binding.btnToggleMode.text = "Switch to Camera Mode"
                    binding.previewView.visibility      = android.view.View.GONE
                    binding.boundingBoxView.visibility  = android.view.View.GONE
                    binding.statsText.text = "Screen mode — switch to your game"

                    ContextCompat.startForegroundService(
                        this,
                        Intent(this, OverlayService::class.java)
                            .setAction(OverlayService.ACTION_START)
                            .putExtra(OverlayService.EXTRA_RESULT_CODE, resultCode)
                            .putExtra(OverlayService.EXTRA_RESULT_DATA, data)
                            .putExtra(OverlayService.EXTRA_CONFIDENCE,  confidence)
                            .putExtra(OverlayService.EXTRA_MODEL,        currentModelFile())
                            .putExtra(OverlayService.EXTRA_PERSON_ONLY,  personOnly),
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
            val preview  = Preview.Builder().build().also {
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
            } catch (e: Exception) { Log.e(TAG, "Camera bind failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(proxy: ImageProxy) {
        if (screenMode || detector == null) { proxy.close(); return }
        val bmp: Bitmap
        try {
            bmp = Bitmap.createBitmap(proxy.width, proxy.height, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(proxy.planes[0].buffer)
        } finally { proxy.close() }

        inferenceScope.launch {
            val t0   = System.currentTimeMillis()
            val dets = runCatching {
                if (debugMode) detector!!.debugTopScores(bmp, swapRB = true)
                else detector!!.detect(bmp, swapRB = true)
            }.getOrDefault(emptyList())
            val ms   = System.currentTimeMillis() - t0
            frameCount++
            avgLatencyMs += (ms - avgLatencyMs) / frameCount
            val fps = 1000.0 / (System.currentTimeMillis() - lastFrameMs).coerceAtLeast(1)
            lastFrameMs = System.currentTimeMillis()
            withContext(Dispatchers.Main) {
                binding.boundingBoxView.setDetections(dets, bmp.width, bmp.height)
                val model = if (useQualityModel) "640" else "320"
                val suffix = when {
                    debugMode  -> " DEBUG"
                    personOnly -> " person"
                    else       -> ""
                }
                val topScore = dets.maxOfOrNull { it.confidence } ?: 0f
                binding.statsText.text =
                    "FPS:%.1f  Lat:%dms  Obj:%d  top:%.0f%%  [%s%s]".format(
                        fps, ms, dets.size, topScore * 100, model, suffix,
                    )
            }
        }
    }

    // -------------------------------------------------------------------------
    // Detector
    // -------------------------------------------------------------------------

    private fun currentModelFile() = if (useQualityModel) MODEL_QUALITY else MODEL_FAST

    private fun loadDetector() {
        val modelFile = currentModelFile()
        inferenceScope.launch {
            detector?.close()
            detector = null
            try {
                val d = YoloDetector(
                    applicationContext,
                    modelFileName        = modelFile,
                    confidenceThreshold  = confidence,
                    personOnly           = personOnly,
                )
                d.load()
                detector = d
                withContext(Dispatchers.Main) {
                    binding.statsText.text =
                        "Ready — ${d.inputSize}×${d.inputSize} model${if (personOnly) " (person only)" else ""}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                withContext(Dispatchers.Main) { binding.statsText.text = "Error: ${e.message}" }
            }
        }
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
