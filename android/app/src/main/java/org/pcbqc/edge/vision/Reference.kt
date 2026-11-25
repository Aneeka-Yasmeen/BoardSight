package org.pcbqc.edge.vision

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import org.pcbqc.edge.core.Box
import org.pcbqc.edge.core.Finding
import org.pcbqc.edge.core.Quality
import org.pcbqc.edge.core.Rules
import kotlin.math.max
import kotlin.math.min

object Reference {
    private const val SIDE = 2400.0
    private const val MIN_INLIERS = 40
    private const val THRESH = 26.0
    private const val TOL = 5.0
    private const val MIN_AREA = 40.0
    private const val MAX_FINDINGS = 25
    private const val EDGE = 17.0
    private const val PLACE_INLIERS = 15

    class Result(val findings: List<Finding>, val note: String, val aligned: Boolean, val toReference: ((Float, Float) -> Pair<Float, Float>)? = null)

    private class Pool {
        private val all = ArrayList<Mat>()
        fun mat() = Mat().also { all += it }
        fun <T : Mat> keep(m: T): T = m.also { all += it }
        fun free() = all.forEach { it.release() }
    }

    private fun gray(b: Bitmap): Pair<Mat, Double> {
        val rgba = Mat()
        Utils.bitmapToMat(b, rgba)
        val g = Mat()
        Imgproc.cvtColor(rgba, g, Imgproc.COLOR_RGBA2GRAY)
        rgba.release()
        val s = min(1.0, SIDE / max(b.width, b.height))
        if (s < 1.0) Imgproc.resize(g, g, Size(), s, s, Imgproc.INTER_AREA)
        return g to s
    }

    private fun flat(g: Mat): Mat {
        val p = Pool()
        try {
            val f = p.mat()
            g.convertTo(f, CvType.CV_32F)
            Imgproc.GaussianBlur(f, f, Size(0.0, 0.0), 1.0)
            val small = Size((g.cols() / 8).toDouble(), (g.rows() / 8).toDouble())
            val mean = p.mat()
            val low = p.mat()
            Imgproc.resize(f, low, small, 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.GaussianBlur(low, low, Size(0.0, 0.0), 5.0)
            Imgproc.resize(low, mean, g.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            Core.subtract(f, mean, f)
            Core.multiply(f, f, mean)
            Imgproc.resize(mean, low, small, 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.GaussianBlur(low, low, Size(0.0, 0.0), 7.5)
            Core.sqrt(low, low)
            Core.max(low, Scalar(12.0), low)
            Imgproc.resize(low, mean, g.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            Core.divide(f, mean, f, 45.0)
            val out = Mat()
            f.convertTo(out, CvType.CV_8U, 1.0, 128.0)
            return out
        } finally {
            p.free()
        }
    }

    fun compare(reference: Bitmap, test: Bitmap): Result {
        val p = Pool()
        try {
            val (g0r, gs) = gray(reference)
            val g0 = p.keep(g0r)
            val (t0r, ts) = gray(test)
            val t0 = p.keep(t0r)
            val g = p.keep(flat(g0))
            val t = p.keep(flat(t0))
            val orb = ORB.create(4000)
            val none = p.mat()
            val kg = p.keep(MatOfKeyPoint())
            val kt = p.keep(MatOfKeyPoint())
            val dg = p.mat()
            val dt = p.mat()
            orb.detectAndCompute(g, none, kg, dg)
            orb.detectAndCompute(t, none, kt, dt)
            if (dg.empty() || dt.empty()) return Result(emptyList(), "reference not comparable (no detail in image)", false)
            val knn = ArrayList<MatOfDMatch>()
            BFMatcher.create(Core.NORM_HAMMING, false).knnMatch(dg, dt, knn, 2)
            knn.forEach { p.keep(it) }
            val pg = kg.toArray()
            val pt = kt.toArray()
            val good = knn.mapNotNull { m ->
                val a = m.toArray()
                if (a.size == 2 && a[0].distance < 0.8f * a[1].distance) a[0] else null
            }
            if (good.size < MIN_INLIERS) return Result(emptyList(), "reference not matched (different board or view)", false)
            val src = p.keep(MatOfPoint2f(*good.map { pg[it.queryIdx].pt }.toTypedArray()))
            val dst = p.keep(MatOfPoint2f(*good.map { pt[it.trainIdx].pt }.toTypedArray()))
            val inl = p.mat()
            val hm = p.keep(Calib3d.findHomography(src, dst, Calib3d.RANSAC, 3.0, inl, 3000, 0.995))
            val n = if (hm.empty()) 0 else Core.countNonZero(inl)
            if (n < MIN_INLIERS) return Result(emptyList(), "reference not matched ($n points agree)", false)

            val inverse = p.keep(hm.inv())
            val h = DoubleArray(9)
            inverse.get(0, 0, h)
            val rw = reference.width * gs
            val rh = reference.height * gs
            val toRef = { x: Float, y: Float ->
                val u = x * ts
                val v = y * ts
                val d = h[6] * u + h[7] * v + h[8]
                Pair(((h[0] * u + h[1] * v + h[2]) / d / rw).toFloat(), ((h[3] * u + h[4] * v + h[5]) / d / rh).toFloat())
            }

            val warped = p.mat()
            Imgproc.warpPerspective(g0, warped, hm, t0.size(), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            val mask = p.mat()
            Imgproc.warpPerspective(p.keep(Mat.ones(g0.size(), CvType.CV_8U)), mask, hm, t0.size(), Imgproc.INTER_NEAREST)
            Imgproc.erode(mask, mask, p.keep(Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(25.0, 25.0))))
            val flags = ByteArray(inl.rows())
            inl.get(0, 0, flags)
            val on = good.indices.filter { flags[it].toInt() != 0 }.map { pt[good[it].trainIdx].pt }
            val hullIdx = p.keep(MatOfInt())
            Imgproc.convexHull(p.keep(MatOfPoint(*on.toTypedArray())), hullIdx)
            val hull = p.keep(MatOfPoint(*hullIdx.toArray().map { on[it] }.toTypedArray()))
            val area = p.keep(Mat.zeros(t0.size(), CvType.CV_8U))
            Imgproc.fillConvexPoly(area, hull, Scalar(1.0))
            Imgproc.dilate(area, area, p.keep(Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(EDGE, EDGE))))
            Core.bitwise_and(mask, area, mask)
            val w = p.keep(flat(warped))
            val k = p.keep(Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(TOL, TOL)))
            val hi = p.mat()
            Imgproc.dilate(w, hi, k)
            Core.subtract(t, hi, hi)
            val lo = p.mat()
            Imgproc.erode(w, lo, k)
            Core.subtract(lo, t, lo)
            Core.max(hi, lo, hi)
            val bin = p.mat()
            Imgproc.threshold(hi, bin, THRESH, 255.0, Imgproc.THRESH_BINARY)
            Core.compare(mask, Scalar(0.0), mask, Core.CMP_EQ)
            bin.setTo(Scalar(0.0), mask)
            Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, p.keep(Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))))
            Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, p.keep(Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(9.0, 9.0))))
            val cnt = ArrayList<MatOfPoint>()
            Imgproc.findContours(bin, cnt, p.mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            cnt.forEach { p.keep(it) }
            val inv = (1 / ts).toFloat()
            val pad = 6f
            val found = cnt.filter { Imgproc.contourArea(it) >= MIN_AREA }.map { c ->
                val r = Imgproc.boundingRect(c)
                val strength = Core.mean(p.keep(hi.submat(r)), p.keep(bin.submat(r))).`val`[0]
                Finding(
                    "anomaly",
                    (strength / 120.0).coerceIn(0.2, 0.95).toFloat(),
                    Box(
                        max(0f, r.x - pad) * inv, max(0f, r.y - pad) * inv,
                        min(test.width.toFloat(), (r.x + r.width + pad) * inv), min(test.height.toFloat(), (r.y + r.height + pad) * inv),
                    ),
                    "reference",
                )
            }.sortedByDescending { it.confidence }
            if (found.size > MAX_FINDINGS) return Result(emptyList(), "reference ignored (${found.size} differences: lighting or focus differs too much)", false, toRef)
            return Result(found, "reference matched ($n points), ${found.size} difference(s)", true, toRef)
        } finally {
            p.free()
        }
    }

    fun samePlace(patch: Bitmap, closeUp: Bitmap): Boolean {
        val p = Pool()
        try {
            val a = p.keep(gray(Bitmap.createScaledBitmap(patch, 640, 640, true)).first)
            val b = p.keep(gray(closeUp).first)
            val orb = ORB.create(1500)
            val none = p.mat()
            val ka = p.keep(MatOfKeyPoint())
            val kb = p.keep(MatOfKeyPoint())
            val da = p.mat()
            val db = p.mat()
            orb.detectAndCompute(a, none, ka, da)
            orb.detectAndCompute(b, none, kb, db)
            if (da.empty() || db.empty()) return false
            val knn = ArrayList<MatOfDMatch>()
            BFMatcher.create(Core.NORM_HAMMING, false).knnMatch(da, db, knn, 2)
            knn.forEach { p.keep(it) }
            val good = knn.mapNotNull { m -> m.toArray().let { if (it.size == 2 && it[0].distance < 0.8f * it[1].distance) it[0] else null } }
            if (good.size < PLACE_INLIERS) return false
            val pa = ka.toArray()
            val pb = kb.toArray()
            val inl = p.mat()
            val h = p.keep(
                Calib3d.findHomography(
                    p.keep(MatOfPoint2f(*good.map { pa[it.queryIdx].pt }.toTypedArray())), p.keep(MatOfPoint2f(*good.map { pb[it.trainIdx].pt }.toTypedArray())),
                    Calib3d.RANSAC, 5.0, inl, 2000, 0.99,
                ),
            )
            return !h.empty() && Core.countNonZero(inl) >= PLACE_INLIERS
        } finally {
            p.free()
        }
    }

    fun quality(b: Bitmap, fromCamera: Boolean): Quality {
        val (g, s) = gray(b)
        val lap = Mat()
        val half = Mat()
        Imgproc.resize(g, half, Size(), 0.5, 0.5, Imgproc.INTER_AREA)
        Imgproc.Laplacian(half, lap, CvType.CV_64F)
        val sd = org.opencv.core.MatOfDouble()
        val mu = org.opencv.core.MatOfDouble()
        Core.meanStdDev(lap, mu, sd)
        val sharp = (sd.toArray()[0] * sd.toArray()[0]).toFloat()
        val mean = Core.mean(g).`val`[0].toFloat()
        val hot = Mat()
        Imgproc.threshold(g, hot, 250.0, 255.0, Imgproc.THRESH_BINARY)
        val glare = 100f * Core.countNonZero(hot) / (g.rows() * g.cols())
        listOf(g, lap, half, hot).forEach { it.release() }
        val p = buildList {
            if (sharp < 40f) add("image is blurred")
            if (mean < Rules.DARK) add("image is too dark")
            if (mean > 225f) add("image is too bright")
            if (fromCamera && glare > 12f) add("too much glare")
            if (max(b.width, b.height) < 400) add("image is too small")
        }
        return Quality(sharp, mean, glare, p, fromCamera && mean < Rules.DARK)
    }
}
