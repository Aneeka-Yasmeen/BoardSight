package org.pcbqc.edge.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.pcbqc.edge.Engine
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

@Composable
fun InspectScreen(m: EdgeModel) {
    val ctx = LocalContext.current
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { u -> u?.let { m.inspectUri(it) } }
    var shoot by remember { mutableStateOf<((Boolean) -> Unit)?>(null) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var torch by remember { mutableStateOf(false) }
    var zoom by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(m.capture.value) { if (m.capture.value > 0 && m.auto) shoot?.invoke(false) }

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().weight(1f).background(Color.Black)) {
            if (granted) CameraView(m, { camera = it }, { shoot = it })
            else Column(Modifier.align(Alignment.Center).padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text("The camera is needed to inspect boards live. You can also check a picture or run the demo without it.")
                Button({ ask.launch(Manifest.permission.CAMERA) }, Modifier.padding(top = 12.dp)) { Text("Allow camera") }
            }
            if (granted) Column(Modifier.align(Alignment.TopStart).padding(10.dp).background(Color(0xC0000000), RoundedCornerShape(8.dp)).padding(horizontal = 10.dp, vertical = 6.dp)) {
                val (done, all) = m.liveProgress
                Text("● LIVE  ${"%.1f".format(m.liveFps)} FPS" + if (done < all) ", scanning $done of $all" else "", color = Color(0xFFFF5A4F), fontSize = 13.sp, fontWeight = FontWeight.Bold)
                val v = m.liveVerdict
                if (m.looking == null && m.liveStrange) Text("This does not look like a bare board", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                else if (m.looking == null && v != null) {
                    val names = m.live.groupBy { it.label }.entries.take(4).joinToString(", ") { "${it.key} x${it.value.size}" }
                    Text("Preview: ${if (names.isEmpty()) "nothing found" else names}", color = verdictColor(v.name), fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    if (names.isNotEmpty()) Text("Likely $v. Tap Inspect board to confirm and record.", color = Color.White, fontSize = 12.sp)
                }
            }
            m.looking?.let { f ->
                Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Color(0xD0000000)).padding(12.dp)) {
                    Text("Second look: move close to the ${f.zone} of the board", color = Copper, fontSize = 16.sp)
                    Text("Put the possible ${f.label} inside the box, hold still, then tap Take close-up.", color = Color.White, fontSize = 13.sp)
                }
                val ins = m.result
                if (ins != null) Canvas(Modifier.align(Alignment.TopEnd).padding(10.dp).size(84.dp, (84f * ins.height / ins.width).coerceIn(50f, 130f).dp)) {
                    drawRect(Color(0xB0000000))
                    drawRect(Copper, style = Stroke(2.dp.toPx()))
                    drawCircle(Color(0xFFFF5A4F), 6.dp.toPx(), Offset(f.box.cx / ins.width * size.width, f.box.cy / ins.height * size.height))
                }
            }
            m.busy?.let {
                Row(Modifier.align(Alignment.Center).background(Color(0xD0000000), RoundedCornerShape(10.dp)).padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(color = Copper)
                    Text(it, Modifier.padding(start = 14.dp), color = Color.White)
                }
            }
        }
        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (camera != null) Slider(zoom, { zoom = it; camera?.cameraControl?.setLinearZoom(it) }, Modifier.height(28.dp))
            if (m.looking != null) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button({ shoot?.invoke(false) }, Modifier.weight(2f).height(54.dp), enabled = granted && m.busy == null) { Text("Take close-up", fontSize = 17.sp) }
                    Button({ m.cancelLook() }, Modifier.weight(1f).height(54.dp), enabled = m.busy == null) { Text("Cancel") }
                }
                return@Column
            }
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(m.auto, { m.auto = !m.auto }, { Text("Auto trigger") }, enabled = granted)
                FilterChip(torch, { torch = !torch; camera?.cameraControl?.enableTorch(torch) }, { Text("Light") }, enabled = camera?.cameraInfo?.hasFlashUnit() == true)
                FilterChip(m.hasReference, { shoot?.invoke(true) }, { Text(if (m.hasReference) "Replace reference" else "Set reference") }, enabled = granted && m.busy == null)
                FilterChip(m.conveyor, { m.toggleConveyor() }, { Text("Conveyor demo") })
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button({ shoot?.invoke(false) }, Modifier.weight(2f).height(54.dp), enabled = granted && m.busy == null) { Text("Inspect board", fontSize = 17.sp) }
                Button({ pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }, Modifier.weight(1f).height(54.dp), enabled = m.busy == null) { Text("Picture") }
            }
        }
    }
}

@SuppressLint("ClickableViewAccessibility")
@Composable
private fun CameraView(m: EdgeModel, onCamera: (Camera?) -> Unit, onShoot: (((Boolean) -> Unit)?) -> Unit) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val view = remember { PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    val worker = remember { Executors.newSingleThreadExecutor() }
    var cam by remember { mutableStateOf<Camera?>(null) }
    val tag = remember { android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { isFakeBoldText = true } }
    val back = remember { android.graphics.Paint() }

    DisposableEffect(owner) {
        val main = ContextCompat.getMainExecutor(ctx)
        val future = ProcessCameraProvider.getInstance(ctx)
        var disposed = false
        val ratio = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
        val preview = Preview.Builder().setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(ratio).build()).build()
        val still = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(ratio)
                    .setResolutionStrategy(ResolutionStrategy(Size(4000, 3000), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)).build(),
            ).build()
        val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(ratio)
                    .setResolutionStrategy(ResolutionStrategy(Size(1600, 1200), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build(),
            ).build()
        analysis.setAnalyzer(worker) { p ->
            try {
                if (m.busy == null && (m.result == null || m.looking != null)) {
                    val f = m.engine.upright(p.toBitmap(), p.imageInfo.rotationDegrees, 1600)
                    m.onFrame(f)
                    f.recycle()
                }
            } catch (e: Throwable) {
                m.viewMessage("Live view error: ${e.message}")
            } finally {
                p.close()
            }
        }
        future.addListener({
            if (disposed) return@addListener
            try {
                val provider = future.get()
                val sel = if (provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
                provider.unbindAll()
                preview.surfaceProvider = view.surfaceProvider
                cam = provider.bindToLifecycle(owner, sel, preview, analysis, still)
                onCamera(cam)
                onShoot { asReference ->
                    still.takePicture(
                        worker,
                        object : ImageCapture.OnImageCapturedCallback() {
                            override fun onCaptureSuccess(image: ImageProxy) {
                                val b = try {
                                    m.engine.upright(image.toBitmap(), image.imageInfo.rotationDegrees)
                                } catch (e: Throwable) {
                                    m.viewMessage("Capture failed: ${e.message}")
                                    null
                                } finally {
                                    image.close()
                                }
                                if (b != null) main.execute { if (m.looking != null) m.finishLook(b) else if (asReference) m.setReference(b) else m.inspect(b, true) }
                            }

                            override fun onError(exception: ImageCaptureException) = m.viewMessage("Capture failed: ${exception.message}")
                        },
                    )
                }
            } catch (e: Throwable) {
                m.viewMessage("Camera could not start: ${e.message}")
            }
        }, main)
        onDispose {
            disposed = true
            onCamera(null)
            onShoot(null)
            runCatching { future.get().unbindAll() }
            worker.shutdown()
        }
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView({ view }, Modifier.fillMaxSize())
        Canvas(
            Modifier.fillMaxSize().pointerInput(cam) {
                detectTapGestures { o ->
                    val pt = view.meteringPointFactory.createPoint(o.x, o.y)
                    cam?.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(pt).build())
                }
            },
        ) {
            val (fw, fh) = m.liveFrame
            if (fw < 2) return@Canvas
            val s = min(size.width / fw, size.height / fh)
            val ox = (size.width - fw * s) / 2
            val oy = (size.height - fh * s) / 2
            val side = min(fw, fh) / 2f * s
            if (m.looking != null) drawRect(
                Color(0x80E0A458), Offset(size.width / 2 - side / 2, size.height / 2 - side / 2),
                androidx.compose.ui.geometry.Size(side, side), style = Stroke(2.dp.toPx()),
            )
            m.live.forEach { f ->
                val b = f.box
                val col = Color(Engine.severityColor(f.severity))
                tag.color = android.graphics.Color.WHITE
                tag.textSize = 12.sp.toPx()
                back.color = Engine.severityColor(f.severity)
                val tx = ox + b.x1 * s
                val ty = max(oy + 16.dp.toPx(), oy + b.y1 * s - 4.dp.toPx())
                drawContext.canvas.nativeCanvas.drawRect(tx, ty - 14.sp.toPx(), tx + tag.measureText(f.label) + 8.dp.toPx(), ty + 4.dp.toPx(), back)
                drawContext.canvas.nativeCanvas.drawText(f.label, tx + 4.dp.toPx(), ty, tag)
                drawRect(
                    col, Offset(ox + b.x1 * s, oy + b.y1 * s),
                    androidx.compose.ui.geometry.Size(max(2f, b.w * s), max(2f, b.h * s)), style = Stroke(2.dp.toPx()),
                )
            }
        }
    }
}
