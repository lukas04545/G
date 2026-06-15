package com.yolodetect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.nio.FloatBuffer

class YoloDetector(
    private val context: Context,
    val modelFileName: String      = "yolov8n.onnx",
    val confidenceThreshold: Float = 0.5f,
    val iouThreshold: Float        = 0.45f,
    val personOnly: Boolean        = false,
) : AutoCloseable {

    companion object {
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
        val shape = (session.inputInfo[inputName]!!.info as TensorInfo).shape
        inputSize = shape[2].toInt()
    }

    // -------------------------------------------------------------------------
    // Inference
    // -------------------------------------------------------------------------

    fun debugTopScores(bitmap: Bitmap, swapRB: Boolean = false): List<Detection> {
        val (tensor, lb) = preprocess(bitmap, swapRB)
        val result = session.run(mapOf(inputName to tensor))
        val raw    = (result[0].value as Array<*>)[0] as Array<*>
        val numFeat = raw.size
        val numDet  = (raw[0] as FloatArray).size

        val bestPerClass = mutableMapOf<Int, Pair<Float, FloatArray>>()
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

        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()
        return bestPerClass.entries
            .sortedByDescending { it.value.first }
            .take(10)
            .map { (cls, pair) ->
                val (score, box) = pair
                val x1 = ((box[0] - box[2] / 2f) - lb.padX) / lb.scale
                val y1 = ((box[1] - box[3] / 2f) - lb.padY) / lb.scale
                val x2 = ((box[0] + box[2] / 2f) - lb.padX) / lb.scale
                val y2 = ((box[1] + box[3] / 2f) - lb.padY) / lb.scale
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

    /**
     * Tiles the image along its long axis and runs [detect] on each tile, then merges
     * results with a global NMS pass. For a tall phone screen this gives ~2× better
     * effective resolution compared to squishing the full screen into one 640-px square.
     */
    fun detectTiled(bitmap: Bitmap, swapRB: Boolean = false): List<Detection> {
        val W = bitmap.width
        val H = bitmap.height
        val tileSize = minOf(W, H)
        val isPortrait = H > W
        val longLen = if (isPortrait) H else W

        if (longLen <= tileSize) return detect(bitmap, swapRB)

        // 2 tiles for ratio ≤ 2.5, 3 tiles for taller screens
        val nTiles = if (longLen.toFloat() / tileSize <= 2.5f) 2 else 3
        val allDets = mutableListOf<Detection>()

        for (ti in 0 until nTiles) {
            val start = ((longLen - tileSize) * ti.toFloat() / (nTiles - 1)).toInt()
            val tile = if (isPortrait)
                Bitmap.createBitmap(bitmap, 0, start, W, tileSize)
            else
                Bitmap.createBitmap(bitmap, start, 0, tileSize, H)

            val offX = if (isPortrait) 0f else start.toFloat()
            val offY = if (isPortrait) start.toFloat() else 0f
            detect(tile, swapRB).forEach { det ->
                allDets += det.copy(bbox = RectF(
                    det.bbox.left  + offX, det.bbox.top    + offY,
                    det.bbox.right + offX, det.bbox.bottom + offY,
                ))
            }
            tile.recycle()
        }

        return nms(allDets)
    }

    fun detect(bitmap: Bitmap, swapRB: Boolean = false): List<Detection> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        val (tensor, lb) = preprocess(bitmap, swapRB)
        val result = session.run(mapOf(inputName to tensor))

        // Output shape: [1, 84, N]
        val raw     = (result[0].value as Array<*>)[0] as Array<*>
        val numFeat = raw.size
        val numDet  = (raw[0] as FloatArray).size

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

            // Undo letterbox: subtract padding, divide by scale
            val x1 = ((cx - w / 2f) - lb.padX) / lb.scale
            val y1 = ((cy - h / 2f) - lb.padY) / lb.scale
            val x2 = ((cx + w / 2f) - lb.padX) / lb.scale
            val y2 = ((cy + h / 2f) - lb.padY) / lb.scale

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

    private data class Letterbox(val scale: Float, val padX: Float, val padY: Float)

    private fun preprocess(bitmap: Bitmap, swapRB: Boolean): Pair<OnnxTensor, Letterbox> {
        val origW  = bitmap.width
        val origH  = bitmap.height
        val scale  = minOf(inputSize.toFloat() / origW, inputSize.toFloat() / origH)
        val fitW   = (origW * scale).toInt()
        val fitH   = (origH * scale).toInt()
        val padX   = (inputSize - fitW) / 2f
        val padY   = (inputSize - fitH) / 2f

        val scaled  = Bitmap.createScaledBitmap(bitmap, fitW, fitH, true)
        val pixels  = IntArray(fitW * fitH)
        scaled.getPixels(pixels, 0, fitW, 0, 0, fitW, fitH)
        if (scaled !== bitmap) scaled.recycle()

        val n   = inputSize * inputSize
        val buf = FloatBuffer.allocate(3 * n)  // zero-filled → black padding

        val padXi = padX.toInt()
        val padYi = padY.toInt()
        for (row in 0 until fitH) {
            for (col in 0 until fitW) {
                val px  = pixels[row * fitW + col]
                // CameraX RGBA buffers land in ARGB_8888 with R and B swapped (swapRB=true).
                // MediaProjection BGRA buffers map correctly — no swap needed (swapRB=false).
                val dst = (row + padYi) * inputSize + (col + padXi)
                if (swapRB) {
                    buf.put(dst,         (px          and 0xFF) / 255f)  // R ← bits 0-7
                    buf.put(n + dst,     ((px shr 8)  and 0xFF) / 255f)  // G
                    buf.put(2 * n + dst, ((px shr 16) and 0xFF) / 255f)  // B ← bits 16-23
                } else {
                    buf.put(dst,         ((px shr 16) and 0xFF) / 255f)  // R ← bits 16-23
                    buf.put(n + dst,     ((px shr 8)  and 0xFF) / 255f)  // G
                    buf.put(2 * n + dst, (px          and 0xFF) / 255f)  // B ← bits 0-7
                }
            }
        }

        val tensor = OnnxTensor.createTensor(
            env, buf,
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        )
        return tensor to Letterbox(scale, padX, padY)
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
