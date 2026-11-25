package org.pcbqc.edge.ui

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.pcbqc.edge.Engine
import org.pcbqc.edge.core.Drift
import org.pcbqc.edge.core.Reasons
import org.pcbqc.edge.core.SecondLook
import org.pcbqc.edge.core.Verdict
import org.pcbqc.edge.data.LogRow

val Copper = Color(0xFFE0A458)
val Panel = Color(0xFF18211E)
val Ground = Color(0xFF0F1513)

private val scheme = darkColorScheme(
    primary = Copper, onPrimary = Color(0xFF2A1A00), background = Ground, surface = Ground, surfaceContainer = Panel,
    onBackground = Color(0xFFE6ECE8), onSurface = Color(0xFFE6ECE8), onSurfaceVariant = Color(0xFFA9B6AF),
    secondaryContainer = Color(0xFF3A2E1B), onSecondaryContainer = Copper, outline = Color(0xFF4A5952),
)

fun verdictColor(v: String) = when (v) {
    "PASS" -> Color(0xFF2E9E5B)
    "REJECT" -> Color(0xFFE53935)
    "REWORK" -> Color(0xFFEF7C00)
    "REVIEW" -> Color(0xFFBE9400)
    else -> Color(0xFF5E7C99)
}

@Composable
fun EdgeScreen(m: EdgeModel) {
    MaterialTheme(colorScheme = scheme) {
        val bar = remember { SnackbarHostState() }
        LaunchedEffect(m.message) {
            m.message?.let {
                m.message = null
                bar.showSnackbar(it)
            }
        }
        Scaffold(
            containerColor = Ground,
            snackbarHost = { SnackbarHost(bar) },
            bottomBar = {
                NavigationBar(containerColor = Panel) {
                    listOf("Inspect", "Log", "Benchmark", "Setup").forEachIndexed { i, t ->
                        NavigationBarItem(
                            selected = m.tab == i,
                            onClick = {
                                m.tab = i
                                if (i == 1) m.refreshLog()
                                if (i == 3) m.refreshMemory()
                            },
                            icon = { Text(listOf("◎", "≡", "▮", "⚙")[i], fontSize = 18.sp) },
                            label = { Text(t) },
                        )
                    }
                }
            },
        ) { pad ->
            Box(Modifier.fillMaxSize().padding(pad).imePadding()) {
                when (m.tab) {
                    0 -> if (m.result != null && m.looking == null) ResultScreen(m) else InspectScreen(m)
                    1 -> LogScreen(m)
                    2 -> BenchScreen(m)
                    else -> SetupScreen(m)
                }
            }
        }
    }
}

@Composable
fun Card(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Surface(color = Panel, shape = RoundedCornerShape(12.dp), modifier = modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) { content() }
    }
}

@Composable
fun Small(t: String) = Text(t, color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)

@Composable
fun ResultScreen(m: EdgeModel) {
    val ins = m.result ?: return
    val why = remember(ins) { ins.why }
    var showWhy by remember(ins) { mutableStateOf(false) }
    var note by remember(ins.boardId) { mutableStateOf("") }
    var choosing by remember(ins.boardId) { mutableStateOf(false) }
    BackHandler { m.closeResult() }
    var zoom by remember(m.picture) { mutableFloatStateOf(1f) }
    var off by remember(m.picture) { mutableStateOf(Offset.Zero) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Surface(color = verdictColor(ins.verdict.name), shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(ins.verdict.name, fontSize = 30.sp, fontWeight = FontWeight.Black, color = Color.White)
                    Spacer(Modifier.weight(1f))
                    Text(ins.boardId, color = Color.White, fontFamily = FontFamily.Monospace)
                }
                Text(why.code, color = Color.White, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                Text(why.short, color = Color.White)
                TextButton({ showWhy = !showWhy; m.hold() }) { Text(if (showWhy) "Hide why" else "Why?", color = Color.White, fontWeight = FontWeight.Bold) }
                if (showWhy) why.details.forEach { Text("• $it", color = Color.White, fontSize = 13.sp, modifier = Modifier.padding(top = 2.dp)) }
            }
        }
        if (m.lookNote.isNotBlank()) Card { Text(m.lookNote) }
        if (ins.notBoard) Card {
            Text("Is this a board?", fontWeight = FontWeight.Bold, color = Copper)
            Small("The app judged nothing because the picture does not look like a bare board. If it is one, continue.")
            Button({ m.inspectAnyway() }) { Text("Inspect anyway") }
        }
        SecondLook.target(ins)?.let { f ->
            if (m.lookNote.isBlank()) Card {
                Text("Not sure about the ${f.label} at the ${f.zone}", fontWeight = FontWeight.Bold, color = Copper)
                Small("A close-up can settle it.")
                Button({ m.startLook() }) { Text("Take a second look") }
            }
        }
        m.picture?.let { p ->
            Box(
                Modifier.fillMaxWidth().heightIn(max = 400.dp).clip(RoundedCornerShape(12.dp)).background(Color.Black)
                    .pointerInput(p) {
                        awaitEachGesture {
                            awaitFirstDown(requireUnconsumed = false)
                            do {
                                val e = awaitPointerEvent()
                                if (e.changes.count { it.pressed } >= 2) {
                                    zoom = (zoom * e.calculateZoom()).coerceIn(1f, 8f)
                                    off = if (zoom == 1f) Offset.Zero else off + e.calculatePan()
                                    e.changes.forEach { if (it.positionChanged()) it.consume() }
                                }
                            } while (e.changes.any { it.pressed })
                        }
                    },
                contentAlignment = Alignment.Center,
            ) {
                Image(
                    p.asImageBitmap(), "Inspected board", contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxWidth().graphicsLayer(scaleX = zoom, scaleY = zoom, translationX = off.x, translationY = off.y),
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Small("Two fingers to zoom and move, one finger to scroll")
                if (zoom > 1f) TextButton({ zoom = 1f; off = Offset.Zero }) { Text("Reset") }
            }
        }
        Card {
            Text("Advice", fontWeight = FontWeight.Bold, color = Copper)
            Text(m.adviceText.ifBlank { if (m.advising) "Llama is reading the findings" else "" })
            Spacer(Modifier.height(6.dp))
            val a = m.advice
            if (a == null) Small("Writing")
            else {
                Small(a.note)
                a.stats?.let { Small("First token ${it.ttftMs} ms, ${"%.1f".format(it.decodeTps)} tokens/s, total ${it.totalMs} ms, ${it.generatedTokens} tokens") }
            }
        }
        if (ins.findings.isNotEmpty() && !ins.notBoard) Card {
            Text("Findings (${ins.findings.size})", fontWeight = FontWeight.Bold, color = Copper)
            ins.findings.take(30).forEach { f ->
                Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.width(6.dp).height(34.dp).background(Color(Engine.severityColor(f.severity)), RoundedCornerShape(3.dp)))
                    Column(Modifier.padding(start = 10.dp)) {
                        Text("${f.label}, ${f.severity.name.lowercase()}, ${f.zone}")
                        Small("${f.reason}. Certainty ${(f.confidence * 100).toInt()}%, found by ${f.source}")
                    }
                }
            }
        }
        if (m.llamaReady && ins.findings.isNotEmpty() && !ins.notBoard) Card {
            var q by remember { mutableStateOf("") }
            Text("Ask Llama about this board", fontWeight = FontWeight.Bold, color = Copper)
            OutlinedTextField(q, { q = it.take(300) }, Modifier.fillMaxWidth(), placeholder = { Text("What should I check first?") }, maxLines = 3)
            TextButton(onClick = { m.ask(q) }, enabled = q.isNotBlank() && !m.asking && !m.advising) { Text(if (m.asking) "Answering" else "Ask") }
            if (m.answer.isNotBlank()) Text(m.answer)
        }
        Card {
            Text("Timing and image", fontWeight = FontWeight.Bold, color = Copper)
            val t = ins.times
            Small("Vision ${Engine.ms(t.totalMs)}: prepare ${Engine.ms(t.prepareMs)}, model ${Engine.ms(t.inferMs)} (${t.tiles} tiles), post ${Engine.ms(t.postMs)}, reference ${Engine.ms(t.goldenMs)}")
            Small("Sharpness ${ins.quality.sharpness.toInt()}, brightness ${ins.quality.brightness.toInt()}, glare ${"%.1f".format(ins.quality.glarePct)}%")
            Small("Reference: ${ins.goldenNote}")
            if (ins.heat.isNotEmpty()) Small("Phone heat: ${ins.heat}")
        }
        if (choosing) Card {
            Text("What should the verdict be?", fontWeight = FontWeight.Bold, color = Copper)
            Small("The app's own verdict stays in the log beside yours.")
            listOf(Verdict.PASS, Verdict.REVIEW, Verdict.REWORK, Verdict.REJECT, Verdict.RETAKE).filter { it != ins.verdict }.forEach { v ->
                Button(
                    { choosing = false; m.operator("overrules", note, v.name) }, Modifier.fillMaxWidth().padding(top = 6.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = verdictColor(v.name), contentColor = Color.White),
                ) { Text(v.name) }
            }
        }
        OutlinedTextField(note, { note = it.take(80) }, Modifier.fillMaxWidth().onFocusChanged { if (it.isFocused) m.hold() }, label = { Text("Operator note (optional, logged with your decision)") }, singleLine = true)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton({ choosing = false; m.operator("agrees", note, ins.verdict.name) }, Modifier.weight(1f)) { Text("Agree") }
            OutlinedButton({ choosing = !choosing; m.hold() }, Modifier.weight(1f)) { Text("Overrule") }
            Button({ if (m.conveyor) m.toggleConveyor(); m.closeResult() }, Modifier.weight(1f)) { Text("Next") }
        }
    }
}

@Composable
fun LogScreen(m: EdgeModel) {
    val save = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { u: Uri? -> u?.let { m.writeText(it, m.engine.db.csv()) } }
    var open by remember { mutableStateOf<Long?>(null) }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        item {
            Row(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton({ save.launch("boardsight_log.csv") }, Modifier.weight(1f), enabled = m.rows.isNotEmpty()) { Text("Save CSV") }
                OutlinedButton({ m.shift() }, Modifier.weight(1f), enabled = !m.asking) { Text("Shift note") }
                OutlinedButton({ m.clearLog() }, Modifier.weight(1f), enabled = m.rows.isNotEmpty()) { Text("Clear") }
            }
        }
        if (m.shiftNote.isNotBlank()) item { Card { Text("Shift note by Llama", color = Copper, fontWeight = FontWeight.Bold); Text(m.shiftNote) } }
        if (m.reasonCounts.isNotEmpty()) item {
            var keys by remember { mutableStateOf(false) }
            Card {
                Text("Boards by reason", color = Copper, fontWeight = FontWeight.Bold)
                m.reasonCounts.forEach { (code, n) -> Text("$code: $n", fontFamily = FontFamily.Monospace, fontSize = 13.sp) }
                TextButton({ keys = !keys }) { Text(if (keys) "Hide code list" else "What do the codes mean?") }
                if (keys) Small(Reasons.legend.joinToString("\n") { "${it.first}  ${it.second}" })
            }
        }
        m.drift?.let { d -> if (d.boards > 0) item { DriftCard(d) } }
        if (m.rows.isEmpty()) item { Small("No inspections yet") }
        items(m.rows, key = LogRow::id) { r ->
            Card(Modifier.clickable { open = if (open == r.id) null else r.id }) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Surface(color = verdictColor(r.verdict), shape = RoundedCornerShape(6.dp)) {
                        Text(r.verdict, Modifier.padding(horizontal = 8.dp, vertical = 2.dp), color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                    Text(r.board, Modifier.padding(start = 10.dp), fontFamily = FontFamily.Monospace)
                    Spacer(Modifier.weight(1f))
                    Small(r.time)
                }
                if (r.reasonCode.isNotBlank()) {
                    Text(r.reasonCode, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Copper)
                    Text(r.reason, fontSize = 14.sp)
                }
                Small("${r.defects} defect(s), vision ${Engine.ms(r.visionMs)}" + if (r.operator.isNotBlank()) ", operator ${r.operator}" + (if (r.finalVerdict.isNotBlank() && r.finalVerdict != r.verdict) ", final ${r.finalVerdict}" else "") else "")
                if (r.note.isNotBlank()) Small("Note: ${r.note}")
                if (open == r.id) {
                    Spacer(Modifier.height(6.dp))
                    r.why.forEach { Small("• $it") }
                    Spacer(Modifier.height(6.dp))
                    SelectionContainer { Text(r.advice) }
                    Small(r.adviceSource)
                    SelectionContainer { Text(r.record, fontFamily = FontFamily.Monospace, fontSize = 11.sp) }
                }
            }
        }
        item { Spacer(Modifier.height(12.dp)) }
    }
}

@Composable
fun DriftCard(d: Drift.Report) = Card {
    Text("Drift map, last ${d.boards} boards against this reference", color = Copper, fontWeight = FontWeight.Bold)
    Canvas(Modifier.fillMaxWidth().height(170.dp).padding(vertical = 8.dp).background(Ground, RoundedCornerShape(8.dp))) {
        d.spots.forEach { drawCircle(Color(0x99A9B6AF), 4.dp.toPx(), Offset(it.x * size.width, it.y * size.height)) }
        d.hotspots.forEach { drawCircle(Color(0xFFFF5A4F), 14.dp.toPx(), Offset(it.x * size.width, it.y * size.height), style = Stroke(3.dp.toPx())) }
    }
    d.hotspots.take(3).forEach { Text("${it.type.replace('_', ' ')} at ${it.zone}: ${it.boards} boards") }
    Small(d.hint)
}

@Composable
fun BenchScreen(m: EdgeModel) {
    val save = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { u: Uri? -> u?.let { m.writeText(it, m.benchText) } }
    var llama by remember { mutableStateOf(true) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Card {
            Text("On-device benchmark", fontWeight = FontWeight.Bold, color = Copper)
            Small("Runs the bundled held-out images, then Llama three times. Reports precision, recall, F1, FPS, memory, power, time to first token and tokens per second. Unplug the phone to measure power.")
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(llama, { llama = it })
                Text("Include Llama", Modifier.padding(start = 10.dp))
            }
            Button({ m.benchmark(llama) }) { Text(if (m.benching) "Stop" else "Run") }
        }
        if (m.benchText.isNotBlank()) Card {
            SelectionContainer { Text(m.benchText, fontFamily = FontFamily.Monospace, fontSize = 12.sp, modifier = Modifier.horizontalScroll(rememberScrollState())) }
            if (!m.benching) TextButton({ save.launch("boardsight_benchmark.json") }) { Text("Save JSON") }
        }
    }
}

@Composable
fun SetupScreen(m: EdgeModel) {
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { m.installModel(it) }
    val ref = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { u -> u?.let { m.setReferenceUri(it) } }
    var conf by remember { mutableFloatStateOf(m.engine.confidence) }
    var use by remember { mutableStateOf(m.engine.useLlama) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Card {
            Text("Llama on this phone", fontWeight = FontWeight.Bold, color = Copper)
            Small(if (m.llamaReady) "Installed. Runs through ExecuTorch on the Arm CPU." else "Not installed. Copy the .pte file and tokenizer.model to the phone, then choose both.")
            Small("Free memory now: ${m.engine.advisor.freeMemory() / 1_000_000} MB")
            m.engine.advisor.loadError?.let { Small(it) }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button({ pick.launch(arrayOf("*/*")) }) { Text("Choose files") }
                if (m.llamaReady) OutlinedButton({ m.removeModel() }) { Text("Remove") }
            }
            if (m.setupNote.isNotBlank()) Small(m.setupNote)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(use, { use = it; m.engine.useLlama = it })
                Text("Use Llama for advice", Modifier.padding(start = 10.dp))
            }
        }
        Card {
            Text("Memory", fontWeight = FontWeight.Bold, color = Copper)
            Small(m.memory.ifBlank { "Tap Refresh" })
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button({ m.refreshMemory() }) { Text("Refresh") }
                OutlinedButton({ m.freeLlama() }) { Text("Unload Llama") }
            }
        }
        Card {
            Text("Reference board", fontWeight = FontWeight.Bold, color = Copper)
            Small(if (m.hasReference) "A reference board is set. Each scan is aligned to it and compared." else "None. Set one to catch differences the model has no name for.")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button({ ref.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }) { Text("Choose picture") }
                if (m.hasReference) OutlinedButton({ m.clearReference() }) { Text("Clear") }
            }
        }
        Card {
            Text("Detection threshold", fontWeight = FontWeight.Bold, color = Copper)
            Small("${(conf * 100).toInt()}%. Lower finds more defects and raises more false alarms.")
            Slider(conf, { conf = it }, onValueChangeFinished = { m.engine.confidence = conf }, valueRange = 0.10f..0.70f, colors = androidx.compose.material3.SliderDefaults.colors())
        }
        Card {
            var check by remember { mutableStateOf(m.engine.checkBoard) }
            Text("Board check", fontWeight = FontWeight.Bold, color = Copper)
            Small("Refuses to judge pictures that do not look like a bare board.")
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(check, { check = it; m.engine.checkBoard = it })
                Text("Check that it is a board", Modifier.padding(start = 10.dp))
            }
        }
        Card {
            Text("Privacy", fontWeight = FontWeight.Bold, color = Copper)
            Small("This app has no internet permission and no access to your photos, files or other apps. The only permission is the camera. Pictures and logs stay in its private folder, are never backed up, and are deleted when you uninstall it. Nothing runs after you close it.")
        }
        Card {
            Text("Built with Llama", fontWeight = FontWeight.Bold, color = Copper)
            Small("Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.")
            Small("Detector: Ultralytics YOLO (AGPL-3.0). Test images: PCB-Defect, Mendeley Data 10.17632/vdj74sngvn.1 (CC BY 4.0), DeepPCB (MIT), DsPCBSD+ 10.6084/m9.figshare.24970329 (CC BY 4.0); resized and recompressed.")
            Small("Also uses ExecuTorch, ONNX Runtime, OpenCV and AndroidX.")
        }
    }
}
