package com.yolodetect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.nio.FloatBuffer

/**
 * On-device YOLOv8 detector backed by ONNX Runtime.
 *
 * Supports any input resolution — the size is read from the ONNX model itself
 * after loading, so switching between yolov8n.onnx (640) and yolov8n_fast.onnx
 * (320) just requires changing [modelFileName].
 */
class YoloDetector(
    private val context: Context,
    val modelFileName: String   = "yolov8n.onnx",
    val confidenceThreshold: Float = 0.5f,
    val iouThreshold: Float     = 0.45f,
    val personOnly: Boolean     = false,
) : AutoCloseable {

    companion object {
        // COCO class index for "person"
        const val CLASS_PERSON = 0
        private const val NUM_CLASSES = 80
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    private var inputName: String = "images"
    var inputSize: Int = 640
        private set

    // -------------------------------------------------------------------------
    // Load
    // -------------------------------------------------------------------------

    fun load() {
        val bytes = context.assets.open(modelFileName).readBytes()
        val opts  = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
        session   = env.createSession(bytes, opts)
        inputName = session.inputNames.iterator().next()

        // Read actual input resolution from the model (shape = [1, 3, H, W])
        val shape = (session.inputInfo[inputName]!!.info as TensorInfo).shape
        inputSize = shape[2].toInt()   // H == W for square YOLO models
    }

    // -------------------------------------------------------------------------
    // Inference
    // -------------------------------------------------------------------------

    /**
     * Like [detect] but ignores [confidenceThreshold] — returns the single
     * highest-scoring detection per class so the UI can show raw model scores.
     * Used in debug mode to diagnose "model sees nothing" situations.
     */
    fun debugTopScores(bitmap: Bitmap): List<Detection> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()
        val tensor = preprocess(bitmap)
        val result = session.run(mapOf(inputName to tensor))
        val raw    = (result[0].value as Array<*>)[0] as Array<*>
        val numFeat = raw.size
        val numDet  = (raw[0] as FloatArray).size

        // Best score per class across all anchor points
        val bestPerClass = mutableMapOf<Int, Pair<Float, FloatArray>>() // classId → (score, xywh)
        for (i in 0 until numDet) {
            for (c in 4 until numFeat) {
                val score = (raw[c] as FloatArray)[i]
                val cls   = c - 4
                val prev  = bestPerClass[cls]?.first ?: 0f
                if (score > prev) {
                    bestPerClass[cls] = score to floatArrayOf(
                        (raw[0] as FloatArray)[i], (raw[1] as FloatArray)[i],
                        (raw[2] as FloatArray)[i], (raw[3] as FloatArray)[i],
                    )
                }
            }
        }

        val scaleX = origW / inputSize
        val scaleY = origH / inputSize
        return bestPerClass.entries
            .sortedByDescending { it.value.first }
            .take(10)
            .map { (cls, pair) ->
                val (score, box) = pair
                val x1 = (box[0] - box[2] / 2f) * scaleX
                val y1 = (box[1] - box[3] / 2f) * scaleY
                val x2 = (box[0] + box[2] / 2f) * scaleX
                val y2 = (box[1] + box[3] / 2f) * scaleY
                Detection(
                    bbox       = RectF(x1.coerceAtLeast(0f), y1.coerceAtLeast(0f),
                                       x2.coerceAtMost(origW), y2.coerceAtMost(origH)),
                    confidence = score,
                    classId    = cls,
                    className  = COCO_CLASSES.getOrElse(cls) { "cls$cls" },
                )
            }
            .also { tensor.close(); result.close() }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        val tensor = preprocess(bitmap)
        val result = session.run(mapOf(inputName to tensor))

        // Output shape: [1, 84, N]  where N = (inputSize/8)² + (inputSize/16)² + (inputSize/32)²
        val raw     = (result[0].value as Array<*>)[0] as Array<*>
        val numFeat = raw.size                           // 84
        val numDet  = (raw[0] as FloatArray).size        // 8400 for 640, 2100 for 320

        val detections = mutableListOf<Detection>()

        for (i in 0 until numDet) {
            val cx = (raw[0] as FloatArray)[i]
            val cy = (raw[1] as FloatArray)[i]
            val w  = (raw[2] as FloatArray)[i]
            val h  = (raw[3] as FloatArray)[i]

            var maxScore = 0f
            var classId  = 0
            for (c in 4 until numFeat) {
                val s = (raw[c] as FloatArray)[i]
                if (s > maxScore) { maxScore = s; classId = c - 4 }
            }

            if (maxScore < confidenceThreshold) continue
            if (personOnly && classId != CLASS_PERSON) continue

            val scaleX = origW / inputSize
            val scaleY = origH / inputSize
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
        result.close()
        return nms(detections)
    }

    // -------------------------------------------------------------------------
    // Pre / post-processing
    // -------------------------------------------------------------------------

    private fun preprocess(bitmap: Bitmap): OnnxTensor {
        val scaled  = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val n       = inputSize * inputSize
        val buf     = FloatBuffer.allocate(3 * n)
        val pixels  = IntArray(n)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (i in 0 until n) {
            val px = pixels[i]
            // RGBA bytes from CameraX/ImageReader land in ARGB_8888 as BGRA after
            // copyPixelsFromBuffer, so getPixels() bits 0-7 hold the camera's Red
            // and bits 16-23 hold the camera's Blue — swap them for YOLO (RGB order).
            buf.put(i,           (px          and 0xFF) / 255f)  // R (in bitmap "Blue" slot)
            buf.put(n + i,       ((px shr 8)  and 0xFF) / 255f)  // G (correct)
            buf.put(2 * n + i,   ((px shr 16) and 0xFF) / 255f)  // B (in bitmap "Red" slot)
        }

        return OnnxTensor.createTensor(
            env, buf,
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        )
    }

    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted     = dets.sortedByDescending { it.confidence }
        val suppressed = BooleanArray(sorted.size)
        val kept       = mutableListOf<Detection>()
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept += sorted[i]
            for (j in i + 1 until sorted.size) {
                if (!suppressed[j]
                    && sorted[i].classId == sorted[j].classId
                    && iou(sorted[i].bbox, sorted[j].bbox) > iouThreshold)
                    suppressed[j] = true
            }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iL = maxOf(a.left, b.left);  val iT = maxOf(a.top, b.top)
        val iR = minOf(a.right, b.right); val iB = minOf(a.bottom, b.bottom)
        if (iR <= iL || iB <= iT) return 0f
        val inter = (iR - iL) * (iB - iT)
        return inter / ((a.right - a.left) * (a.bottom - a.top) +
                        (b.right - b.left) * (b.bottom - b.top) - inter)
    }

    override fun close() {
        if (::session.isInitialized) session.close()
    }
}

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
