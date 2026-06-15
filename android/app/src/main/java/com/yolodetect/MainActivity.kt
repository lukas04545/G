package com.yolodetect

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
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
        private const val REQUEST_CAMERA = 10
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: YoloDetector
    private lateinit var cameraExecutor: ExecutorService

    // Inference runs on this scope; UI updates hop to Dispatchers.Main
    private val inferenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var frameWidth  = YoloDetector.INPUT_SIZE
    private var frameHeight = YoloDetector.INPUT_SIZE

    // --- Simple FPS / latency tracking ---
    private var lastFrameMs = System.currentTimeMillis()
    private var avgLatencyMs = 0.0
    private var frameCount = 0

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (hasCameraPermission()) startCamera() else requestCameraPermission()

        cameraExecutor = Executors.newSingleThreadExecutor()
        loadDetector()
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
        cameraExecutor.shutdown()
        if (::detector.isInitialized) detector.close()
    }

    // -------------------------------------------------------------------------
    // Model
    // -------------------------------------------------------------------------

    private fun loadDetector() {
        inferenceScope.launch {
            try {
                detector = YoloDetector(applicationContext)
                detector.load()
                Log.i(TAG, "Model loaded")
                withContext(Dispatchers.Main) {
                    binding.statsText.text = "Model ready — point camera at something"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity,
                        "Model not found. Run export_model.py and rebuild.", Toast.LENGTH_LONG).show()
                    binding.statsText.text = "Error: ${e.message}"
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Camera
    // -------------------------------------------------------------------------

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            androidx.camera.core.resolutionselector.ResolutionStrategy(
                                Size(640, 640),
                                androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            )
                        ).build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // -------------------------------------------------------------------------
    // Analysis
    // -------------------------------------------------------------------------

    private fun analyzeFrame(proxy: ImageProxy) {
        if (!::detector.isInitialized) { proxy.close(); return }

        val bitmap: Bitmap
        try {
            // CameraX guarantees RGBA_8888 because we set OUTPUT_IMAGE_FORMAT_RGBA_8888
            bitmap = Bitmap.createBitmap(proxy.width, proxy.height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(proxy.planes[0].buffer)
        } finally {
            proxy.close()
        }

        frameWidth  = bitmap.width
        frameHeight = bitmap.height

        inferenceScope.launch {
            val t0 = System.currentTimeMillis()
            val detections = runCatching { detector.detect(bitmap) }.getOrDefault(emptyList())
            val elapsed = System.currentTimeMillis() - t0

            // Rolling average latency
            frameCount++
            avgLatencyMs += (elapsed - avgLatencyMs) / frameCount

            val now = System.currentTimeMillis()
            val fps = 1000.0 / (now - lastFrameMs).coerceAtLeast(1)
            lastFrameMs = now

            withContext(Dispatchers.Main) {
                binding.boundingBoxView.setDetections(detections, frameWidth, frameHeight)
                binding.statsText.text =
                    "FPS: %.1f  |  Latency: %dms  |  Objects: %d".format(fps, elapsed, detections.size)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Permissions
    // -------------------------------------------------------------------------

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() =
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == REQUEST_CAMERA && results.firstOrNull() == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }
}
