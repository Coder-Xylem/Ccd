package com.embeddinggemma.offline.ui

import android.graphics.Bitmap
import android.graphics.Color as AColor
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.embeddinggemma.offline.AppState
import com.embeddinggemma.offline.core.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

@Composable
fun ToolsScreen(app: AppState) {
    val names = listOf("Compare", "Classify", "Organize", "Duplicates", "Inspector")
    var t by remember { mutableIntStateOf(0) }
    Column(Modifier.fillMaxSize()) {
        ScrollableTabRow(t, containerColor = MaterialTheme.colorScheme.background, edgePadding = 8.dp) {
            names.forEachIndexed { i, n -> Tab(t == i, { t = i }, text = { Text(n) }) }
        }
        Column(Modifier.padding(16.dp)) {
            ModelBanner(app)
            when (t) { 0 -> Compare(app); 1 -> Classify(app); 2 -> Organize(app); 3 -> DuplicatesTool(app); else -> Inspector(app) }
        }
    }
}

private class Side { var text by mutableStateOf(""); var image by mutableStateOf<ByteArray?>(null) }

@Composable
private fun SideInput(label: String, s: Side, canImage: Boolean) {
    val ctx = LocalContext.current
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { u -> u?.let { s.image = ctx.contentResolver.openInputStream(it)?.use { i -> i.readBytes() } } }
    OutlinedTextField(s.text, { s.text = it }, Modifier.fillMaxWidth(), label = { Text(label) }, enabled = s.image == null, minLines = 2)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        if (canImage) OutlinedButton({ pick.launch("image/*") }) { Text(if (s.image == null) "Use image" else "Image selected") }
        if (s.image != null) TextButton({ s.image = null }) { Text("Clear image") }
    }
}

@Composable
private fun Compare(app: AppState) {
    val scope = rememberCoroutineScope(); val eng by app.engine.collectAsState()
    val a = remember { Side() }; val b = remember { Side() }
    var score by remember { mutableStateOf<Float?>(null) }; var err by remember { mutableStateOf<String?>(null) }
    SideInput("A", a, eng?.info?.image == true); SideInput("B", b, eng?.info?.image == true)
    Button(onClick = {
        val em = app.embedder() ?: return@Button
        scope.launch {
            err = null
            try {
                score = withContext(Dispatchers.Default) {
                    val bothText = a.image == null && b.image == null
                    fun vec(s: Side) = s.image?.let { em.image(it) } ?: em.query(s.text, if (bothText) EmbeddingTask.SENTENCE_SIMILARITY else EmbeddingTask.SEARCH)
                    VectorMath.dot(vec(a), vec(b))
                }
            } catch (t: Throwable) { err = t.message }
        }
    }, enabled = eng != null && (a.image != null || a.text.isNotBlank()) && (b.image != null || b.text.isNotBlank())) { Text("Compare") }
    score?.let { Text(pct(it), style = MaterialTheme.typography.displaySmall, modifier = Modifier.padding(top = 12.dp)); LinearProgressIndicator({ it.coerceIn(0f, 1f) }, Modifier.fillMaxWidth())
        Muted("Cosine similarity. Scores are only comparable within the same kind of pair (text/text, text/image, image/image).") }
    err?.let { Muted(it) }
}

@Composable
private fun Classify(app: AppState) {
    val scope = rememberCoroutineScope(); val eng by app.engine.collectAsState()
    var text by remember { mutableStateOf("") }; var labels by remember { mutableStateOf("Research, College, Personal, Finance, Career") }
    var out by remember { mutableStateOf<List<LabelScore>>(emptyList()) }
    OutlinedTextField(text, { text = it }, Modifier.fillMaxWidth(), label = { Text("Text") }, minLines = 3)
    OutlinedTextField(labels, { labels = it }, Modifier.fillMaxWidth(), label = { Text("Categories (comma separated)") })
    Button(onClick = {
        val em = app.embedder() ?: return@Button
        scope.launch { out = withContext(Dispatchers.Default) {
            val ls = labels.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it to em.query(it, EmbeddingTask.CLASSIFICATION) }
            ZeroShot.rank(em.query(text, EmbeddingTask.CLASSIFICATION), ls)
        } }
    }, enabled = eng != null && text.isNotBlank()) { Text("Classify") }
    out.forEachIndexed { i, s -> Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(s.label, style = if (i == 0) MaterialTheme.typography.titleMedium else MaterialTheme.typography.bodyMedium); Text("sim ${pct(s.similarity)} · share ${pct(s.share)}") } }
    if (out.isNotEmpty()) Muted("Similarity is cosine. Share is a softmax across your categories: a relative ranking, not a calibrated probability.")
}

@Composable
private fun Organize(app: AppState) {
    val scope = rememberCoroutineScope(); val version by app.version.collectAsState()
    var coll by remember { mutableStateOf<String?>(null) }; var k by remember { mutableIntStateOf(0) }
    var groups by remember { mutableStateOf<List<Pair<String, List<String>>>>(emptyList()) }; var msg by remember { mutableStateOf<String?>(null) }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Chooser("Collection", listOf<String?>(null) + app.store.collections(), coll, { it ?: "All" }) { coll = it }
        Chooser("Groups", listOf(0, 2, 3, 4, 5, 6, 8), k, { if (it == 0) "Auto" else "$it" }) { k = it }
    }
    Button(onClick = { scope.launch {
        groups = withContext(Dispatchers.Default) {
            val iv = itemVectors(app, coll); val ids = iv.keys.toList()
            if (ids.size < 2) { msg = "Index at least two items first."; emptyList() } else {
                msg = null
                val asg = KMeans.cluster(ids.map { iv.getValue(it) }, if (k == 0) KMeans.suggestK(ids.size) else k)
                asg.indices.groupBy { asg[it] }.values.map { m ->
                    val cen = VectorMath.mean(m.map { iv.getValue(ids[it]) })
                    val sorted = m.sortedByDescending { VectorMath.dot(iv.getValue(ids[it]), cen) }.map { app.store.item(ids[it])!!.name }
                    sorted.first() to sorted
                }.sortedByDescending { it.second.size }
            }
        } } }, modifier = Modifier.padding(top = 8.dp)) { Text("Organize automatically") }
    msg?.let { Muted(it) }
    LazyColumn { items(groups.size) { i -> val (n, ms) = groups[i]
        Title("Group ${i + 1} · like “$n” (${ms.size})"); Text(ms.take(8).joinToString("\n"), color = MaterialTheme.colorScheme.onSurfaceVariant); if (ms.size > 8) Muted("+ ${ms.size - 8} more"); Rule() } }
}

@Composable
private fun DuplicatesTool(app: AppState) {
    val scope = rememberCoroutineScope()
    var th by remember { mutableFloatStateOf(0.92f) }
    var pairs by remember { mutableStateOf<List<DupPair>?>(null) }; var shown by remember { mutableStateOf(setOf<Long>()) }
    var dismissed by remember { mutableStateOf(setOf<Pair<Long, Long>>()) }; var confirm by remember { mutableStateOf<Long?>(null) }
    fun scan() = scope.launch { pairs = withContext(Dispatchers.Default) { Duplicates.find(itemVectors(app, null), th) } }
    Text("Similarity threshold ${pct(th)}"); Slider(th, { th = it }, valueRange = 0.7f..0.99f)
    Button(onClick = { scan() }) { Text("Find near-duplicates") }
    pairs?.filter { (it.a to it.b) !in dismissed }?.let { ps ->
        if (ps.isEmpty()) Muted("No near-duplicates found.")
        LazyColumn { items(ps, key = { "${it.a}-${it.b}" }) { p ->
            val ia = app.store.item(p.a); val ib = app.store.item(p.b)
            if (ia != null && ib != null) Column(Modifier.padding(vertical = 8.dp)) {
                Text("${ia.name}\n${ib.name}"); Muted("${pct(p.similarity)} similar")
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton({ shown = if (p.a in shown) shown - p.a else shown + p.a }) { Text("Compare") }
                    TextButton({ dismissed = dismissed + (p.a to p.b) }) { Text("Keep both") }
                    TextButton({ confirm = p.a }) { Text("Remove A") }; TextButton({ confirm = p.b }) { Text("Remove B") }
                }
                if (p.a in shown) { Muted("A: " + (app.store.chunks(ia.id).firstOrNull()?.text ?: ia.name).take(300)); Muted("B: " + (app.store.chunks(ib.id).firstOrNull()?.text ?: ib.name).take(300)) }
                Rule()
            } } }
    }
    confirm?.let { id -> Confirm("Remove from index?", "This removes the item from the app's index (notes are deleted). Original files on your device are never touched.",
        { app.deleteItem(id); confirm = null; scan() }, { confirm = null }) }
}

@Composable
private fun Inspector(app: AppState) {
    val scope = rememberCoroutineScope(); val eng by app.engine.collectAsState(); val dims by app.dims.collectAsState()
    var text by remember { mutableStateOf("") }; var task by remember { mutableStateOf(EmbeddingTask.SEARCH) }; var asDoc by remember { mutableStateOf(false) }
    var report by remember { mutableStateOf("") }; var bench by remember { mutableStateOf("") }
    OutlinedTextField(text, { text = it }, Modifier.fillMaxWidth(), label = { Text("Input") }, minLines = 2)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Chooser("Task", allTasks, task, { it.label }) { task = it }
        FilterChip(asDoc, { asDoc = !asDoc }, { Text("Document form") })
    }
    Button(onClick = { val em = app.embedder() ?: return@Button; scope.launch { report = withContext(Dispatchers.Default) {
        val fmt = if (asDoc) task.formatDocument(text) else task.formatQuery(text)
        val t0 = System.nanoTime(); val v = if (asDoc) em.document(text, null, task) else em.query(text, task); val ms = (System.nanoTime() - t0) / 1_000_000
        val nrm = "%.4f".format(VectorMath.norm(v))
        "Model: ${em.info.name}\nDimensions: ${v.size} (native ${em.info.nativeDims})\nNorm: $nrm (normalised after truncation)\nTask: ${task.label}\nFormatted input: $fmt\nProcessing: $ms ms\nFirst values: ${v.take(6).joinToString { "%.3f".format(it) }}"
    } } }, enabled = eng != null && text.isNotBlank()) { Text("Embed") }
    if (report.isNotEmpty()) Text(report, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 8.dp))
    Rule(); Title("Local benchmark")
    Button(onClick = { val em = app.embedder() ?: return@Button; scope.launch { bench = "Running…"; bench = withContext(Dispatchers.Default) {
        val rt = Runtime.getRuntime(); val sample = "Semantic search finds meaning, not just matching words. ".repeat(8)
        em.document(sample) // warm-up
        val n = 10; val t0 = System.nanoTime(); repeat(n) { em.document(sample) }; val txt = (System.nanoTime() - t0) / 1e6 / n
        var img = "n/a (no vision encoder)"
        if (em.info.image) { val bm = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888).apply { eraseColor(AColor.GRAY) }
            val bos = ByteArrayOutputStream(); bm.compress(Bitmap.CompressFormat.JPEG, 90, bos); val bytes = bos.toByteArray()
            em.image(bytes); val t1 = System.nanoTime(); repeat(3) { em.image(bytes) }; img = "%.0f ms".format((System.nanoTime() - t1) / 1e6 / 3) }
        "Text (~${sample.length} chars, $dims dims): %.0f ms avg of $n\nImage (224×224): $img\nJVM heap used: ${(rt.totalMemory() - rt.freeMemory()) shr 20} MB\nAudio / video: not benchmarked in this build.\nDevice-dependent: results vary with thermal state and background load.".format(txt)
    } } }, enabled = eng != null) { Text("Run benchmark") }
    if (bench.isNotEmpty()) Text(bench, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 8.dp))
}
