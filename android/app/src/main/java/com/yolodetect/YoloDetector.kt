package com.yolodetect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.math.exp

class YoloDetector(
    private val context: Context,
    val modelFileName: String      = "yolo11n_seg.onnx",
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
        val opts  = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            // XNNPACK: optimised ARM NEON kernels — pure CPU, no data-transfer overhead.
            // Outperforms NNAPI on YOLO because NNAPI can't handle all ops (Reshape/
            // Transpose bounce back to CPU, killing throughput with NNAPI).
            runCatching { addXnnpack(emptyMap()) }
        }
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
            for (c in 4 until minOf(84, numFeat)) {
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
     * effective resolution compared to squishing the full screen into one 320-px square.
     */
    fun detectTiled(bitmap: Bitmap, swapRB: Boolean = false): List<Detection> {
        val W = bitmap.width
        val H = bitmap.height
        val tileSize = minOf(W, H)
        val isPortrait = H > W
        val longLen = if (isPortrait) H else W

        if (longLen <= tileSize) return detect(bitmap, swapRB)

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

        val raw     = (result[0].value as Array<*>)[0] as Array<*>
        val numFeat = raw.size
        val numDet  = (raw[0] as FloatArray).size
        // YOLO seg: 4 bbox + 80 classes + 32 mask coefficients = 116 features
        val numMask = (numFeat - 84).coerceAtLeast(0)
        val isSeg   = numMask > 0

        // Flatten prototype masks from [numMask, protoSize, protoSize] into Array<FloatArray>
        val protoSize = inputSize / 4  // 80 for 320px model
        val protos: Array<FloatArray>? = if (isSeg) {
            val proto0 = (result[1].value as Array<*>)[0] as Array<*>
            Array(numMask) { k ->
                val rows = proto0[k] as Array<*>
                FloatArray(protoSize * protoSize) { idx ->
                    (rows[idx / protoSize] as FloatArray)[idx % protoSize]
                }
            }
        } else null

        // Parse anchors: collect detections with mask coefficients before closing result
        val pre = mutableListOf<Pair<Detection, FloatArray?>>()
        for (i in 0 until numDet) {
            val cx = (raw[0] as FloatArray)[i]; val cy = (raw[1] as FloatArray)[i]
            val w  = (raw[2] as FloatArray)[i]; val h  = (raw[3] as FloatArray)[i]

            var maxScore = 0f; var classId = 0
            for (c in 4 until 84) {
                val s = (raw[c] as FloatArray)[i]
                if (s > maxScore) { maxScore = s; classId = c - 4 }
            }
            if (maxScore < confidenceThreshold) continue
            if (personOnly && classId != CLASS_PERSON) continue

            val x1 = ((cx - w / 2f) - lb.padX) / lb.scale
            val y1 = ((cy - h / 2f) - lb.padY) / lb.scale
            val x2 = ((cx + w / 2f) - lb.padX) / lb.scale
            val y2 = ((cy + h / 2f) - lb.padY) / lb.scale

            val coeffs = if (isSeg) FloatArray(numMask) { k -> (raw[84 + k] as FloatArray)[i] } else null

            pre += Pair(
                Detection(
                    bbox       = RectF(x1.coerceAtLeast(0f), y1.coerceAtLeast(0f),
                                       x2.coerceAtMost(origW), y2.coerceAtMost(origH)),
                    confidence = maxScore,
                    classId    = classId,
                    className  = COCO_CLASSES.getOrElse(classId) { "cls$classId" },
                ),
                coeffs,
            )
        }

        tensor.close(); result.close()

        // NMS — inline to keep coefficients paired with detections
        val sorted     = pre.sortedByDescending { it.first.confidence }
        val suppressed = BooleanArray(sorted.size)
        val kept       = mutableListOf<Pair<Detection, FloatArray?>>()
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept += sorted[i]
            for (j in i + 1 until sorted.size) {
                if (!suppressed[j]
                    && sorted[i].first.classId == sorted[j].first.classId
                    && iou(sorted[i].first.bbox, sorted[j].first.bbox) > iouThreshold)
                    suppressed[j] = true
            }
        }

        // Compute masks only for the kept detections
        return kept.map { (det, coeffs) ->
            if (protos != null && coeffs != null)
                det.copy(mask = computeMask(coeffs, protos, lb, det.bbox, protoSize))
            else det
        }
    }

    // -------------------------------------------------------------------------
    // Segmentation mask
    // -------------------------------------------------------------------------

    private fun computeMask(
        coeffs: FloatArray,
        protos: Array<FloatArray>,   // [numMask][protoSize²]
        lb: Letterbox,
        bboxOrig: RectF,
        protoSize: Int,
    ): Bitmap {
        // Map bbox from original-image coords to prototype coords:
        //   orig → model input: x_m = x_o * scale + padX
        //   model input → proto: x_p = x_m * (protoSize / inputSize)
        val mToP = protoSize.toFloat() / inputSize
        val px1 = ((bboxOrig.left   * lb.scale + lb.padX) * mToP).toInt().coerceIn(0, protoSize)
        val py1 = ((bboxOrig.top    * lb.scale + lb.padY) * mToP).toInt().coerceIn(0, protoSize)
        val px2 = ((bboxOrig.right  * lb.scale + lb.padX) * mToP).toInt().coerceIn(0, protoSize)
        val py2 = ((bboxOrig.bottom * lb.scale + lb.padY) * mToP).toInt().coerceIn(0, protoSize)
        val mW  = (px2 - px1).coerceAtLeast(1)
        val mH  = (py2 - py1).coerceAtLeast(1)

        val bmp = Bitmap.createBitmap(mW, mH, Bitmap.Config.ALPHA_8)
        val buf = ByteBuffer.allocate(mW * mH)
        for (y in 0 until mH) {
            for (x in 0 until mW) {
                val pIdx = (py1 + y) * protoSize + (px1 + x)
                var dot = 0f
                for (k in coeffs.indices) dot += coeffs[k] * protos[k][pIdx]
                buf.put(if (sigmoid(dot) > 0.5f) 255.toByte() else 0.toByte())
            }
        }
        buf.rewind()
        bmp.copyPixelsFromBuffer(buf)
        return bmp
    }

    private fun sigmoid(x: Float) = 1f / (1f + exp(-x))

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
