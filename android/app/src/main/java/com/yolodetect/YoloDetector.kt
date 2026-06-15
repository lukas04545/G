package com.yolodetect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.nio.FloatBuffer

/**
 * On-device YOLOv8 object detector backed by ONNX Runtime.
 *
 * Expects the model to be in assets/ and to accept a [1, 3, INPUT_SIZE, INPUT_SIZE]
 * float32 tensor (RGB, values in [0, 1]) and output [1, 84, 8400].
 */
class YoloDetector(
    private val context: Context,
    private val modelFileName: String = "yolov8n.onnx",
    val confidenceThreshold: Float = 0.5f,
    val iouThreshold: Float = 0.45f,
) : AutoCloseable {

    companion object {
        const val INPUT_SIZE = 640
        private const val NUM_CLASSES = 80  // COCO
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    private var inputName: String = "images"

    /** Load the model from assets. Call once before [detect]. */
    fun load() {
        val bytes = context.assets.open(modelFileName).readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.iterator().next()
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /**
     * Run detection on [bitmap] (any size). Returns a list of [Detection]
     * with bounding boxes in the *original* bitmap coordinate space.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        val tensor = preprocess(bitmap)
        val inputs = mapOf(inputName to tensor)
        val output = session.run(inputs)

        // output shape: [1, 84, 8400]
        val raw = (output[0].value as Array<*>)[0] as Array<*>
        val numFeatures = raw.size                           // 84
        val numDets = (raw[0] as FloatArray).size           // 8400

        val detections = mutableListOf<Detection>()
        for (i in 0 until numDets) {
            val cx = (raw[0] as FloatArray)[i]
            val cy = (raw[1] as FloatArray)[i]
            val w  = (raw[2] as FloatArray)[i]
            val h  = (raw[3] as FloatArray)[i]

            var maxScore = 0f
            var classId  = 0
            for (c in 4 until numFeatures) {
                val score = (raw[c] as FloatArray)[i]
                if (score > maxScore) { maxScore = score; classId = c - 4 }
            }

            if (maxScore < confidenceThreshold) continue

            // Convert from 640-space to original image space
            val scaleX = origW / INPUT_SIZE
            val scaleY = origH / INPUT_SIZE
            val x1 = (cx - w / 2f) * scaleX
            val y1 = (cy - h / 2f) * scaleY
            val x2 = (cx + w / 2f) * scaleX
            val y2 = (cy + h / 2f) * scaleY

            detections += Detection(
                bbox       = RectF(x1.coerceAtLeast(0f), y1.coerceAtLeast(0f),
                                   x2.coerceAtMost(origW), y2.coerceAtMost(origH)),
                confidence = maxScore,
                classId    = classId,
                className  = COCO_CLASSES.getOrElse(classId) { "cls$classId" },
            )
        }

        tensor.close()
        output.close()

        return nms(detections)
    }

    // ------------------------------------------------------------------
    // Pre / post-processing
    // ------------------------------------------------------------------

    private fun preprocess(bitmap: Bitmap): OnnxTensor {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val n = INPUT_SIZE * INPUT_SIZE
        val buf = FloatBuffer.allocate(3 * n)
        val r = buf.array(); val g = r; val b = r  // all same backing array — index by offset

        // Fill in CHW order: R plane, G plane, B plane
        for (i in 0 until n) {
            val px = pixels[i]
            buf.put(i,       ((px shr 16) and 0xFF) / 255f)  // R
            buf.put(n + i,   ((px shr 8)  and 0xFF) / 255f)  // G
            buf.put(2 * n + i, (px and 0xFF)        / 255f)  // B
        }

        return OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
    }

    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.confidence }
        val suppressed = BooleanArray(sorted.size)
        val kept = mutableListOf<Detection>()
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept += sorted[i]
            for (j in i + 1 until sorted.size) {
                if (!suppressed[j] && sorted[i].classId == sorted[j].classId &&
                    iou(sorted[i].bbox, sorted[j].bbox) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iLeft   = maxOf(a.left,   b.left)
        val iTop    = maxOf(a.top,    b.top)
        val iRight  = minOf(a.right,  b.right)
        val iBottom = minOf(a.bottom, b.bottom)
        if (iRight <= iLeft || iBottom <= iTop) return 0f
        val inter = (iRight - iLeft) * (iBottom - iTop)
        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        return inter / (aArea + bArea - inter)
    }

    override fun close() {
        if (::session.isInitialized) session.close()
    }
}

// 80 COCO class names in index order
val COCO_CLASSES = listOf(
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
)
