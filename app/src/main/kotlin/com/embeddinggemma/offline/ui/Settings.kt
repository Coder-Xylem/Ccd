package com.embeddinggemma.offline.ui

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Environment
import android.os.StatFs
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.embeddinggemma.offline.AppState
import com.embeddinggemma.offline.data.ModelVariant
import com.embeddinggemma.offline.data.VideoMode

private data class Preset(val name: String, val dims: Int, val note: String)
private val presets = listOf(Preset("Maximum quality", 768, "Best semantic quality, most storage"), Preset("Balanced", 512, "Good quality, about 33% smaller"),
    Preset("Storage saver", 256, "Compact, about 67% smaller"), Preset("Maximum compactness", 128, "Smallest and fastest, lowest quality"))

@Composable
fun SettingsScreen(app: AppState) {
    val ctx = LocalContext.current
    val status by app.modelStatus.collectAsState(); val eng by app.engine.collectAsState(); val dims by app.dims.collectAsState(); val vm by app.videoMode.collectAsState()
    var pendingDims by remember { mutableStateOf<Int?>(null) }; var wipe by remember { mutableStateOf<String?>(null) }
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let(app::importModel) }
    val file = app.modelFile(); val variant = app.variant()

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text("Model Manager", style = MaterialTheme.typography.headlineSmall)
        Title("EmbeddingGemma 2")
        KV("Status", if (file == null) "Not installed" else "Installed · $status")
        KV("File", file?.let { "${it.name} (${it.length() shr 20} MB)" } ?: "–")
        KV("Variant", variant?.let { "${it.label} · ${it.params}" } ?: "–")
        KV("Text / code", if (variant != null) "Enabled" else "–")
        KV("Image / video frames", if (variant?.vision == true) "Enabled" else "Needs text+vision or omnimodal bundle")
        KV("Audio", if (variant?.audioEncoder == true) "Encoder present; not wired in this build" else "Not available")
        KV("Embedding / context", "768 native · 8K tokens")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(vertical = 8.dp)) {
            Button({ importer.launch(arrayOf("*/*")) }) { Text(if (file == null) "Import model" else "Replace model") }
            OutlinedButton({ Thread { app.loadModel() }.start() }, enabled = file != null) { Text("Reload") }
            OutlinedButton({ app.removeModel() }, enabled = file != null) { Text("Remove") }
        }
        Muted("Import a .litertlm bundle from the LiteRT Community on Hugging Face (embeddinggemma-2-text-270m, -text-vision-440m, or -740m). Variant is detected from the file name. Check the model card there for license terms.")

        Title("Embedding size")
        presets.forEach { p -> Row(Modifier.fillMaxWidth().clickable { if (p.dims != dims) pendingDims = p.dims }.padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            RadioButton(dims == p.dims, { if (p.dims != dims) pendingDims = p.dims })
            Column { Text("${p.name} · ${p.dims}"); Muted("${p.note} · ~${p.dims * 4 * 10_000 / 1_000_000} MB per 10,000 chunks") } } }
        Title("Video sampling")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { VideoMode.values().forEach { m -> FilterChip(vm == m, { app.setVideoMode(m) }, { Text("${m.label} (${m.frames})") }) } }
        Muted("Frames per video. Applies to videos indexed from now on.")

        Title("Device check")
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val ramGb = mi.totalMem / 1e9; val freeGb = StatFs(Environment.getDataDirectory().path).availableBytes / 1e9
        KV("RAM", "%.1f GB".format(ramGb)); KV("Free storage", "%.1f GB".format(freeGb)); KV("CPU ABI", Build.SUPPORTED_ABIS.firstOrNull() ?: "?")
        val rec = when { ramGb < 4 -> "256 dimensions · text only (270M)"; ramGb < 6 -> "512 dimensions · text + image (440M)"; else -> "768 or 512 dimensions · any variant" }
        KV("Recommended", rec); Muted("Guidance only. Google reports ~191 MB active RAM for text-only and ~567 MB for the full model on a Pixel 11 Pro; other devices differ.")

        Title("Privacy")
        listOf("AI processing" to "LOCAL", "Embeddings" to "LOCAL", "Vector database" to "LOCAL (SQLite on this device)", "Internet required" to "NO (the app declares no INTERNET permission)", "Cloud AI" to "NO", "Analytics" to "NONE").forEach { KV(it.first, it.second) }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 8.dp)) {
            OutlinedButton({ wipe = "emb" }) { Text("Delete embeddings") }; OutlinedButton({ wipe = "idx" }) { Text("Delete index") }
        }
        OutlinedButton({ wipe = "all" }, Modifier.padding(top = 8.dp)) { Text("Delete all app data") }
        Spacer(Modifier.height(16.dp))
        Muted("EmbeddingGemma 2 is a Google model; this app is an independent client and not affiliated with Google. See the model card for license, intended use and limitations.")
    }
    pendingDims?.let { d -> Confirm("Change to $d dimensions?", "Stored vectors use a fixed size, so everything will be re-indexed with the new size. Your files are not modified.",
        { app.setDims(d); pendingDims = null }, { pendingDims = null }) }
    wipe?.let { w -> Confirm("Are you sure?", when (w) { "emb" -> "Deletes all embeddings. Items stay and are re-indexed."; "idx" -> "Deletes the whole index and your notes. Original files are not touched."; else -> "Deletes the index, notes, history, settings and the imported model." },
        { when (w) { "emb" -> app.deleteEmbeddings(); "idx" -> app.deleteIndexedData(); else -> app.deleteAllAppData() }; wipe = null }, { wipe = null }) }
}

@Composable private fun KV(k: String, v: String) = Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), horizontalArrangement = Arrangement.SpaceBetween) {
    Text(k, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(0.4f)); Text(v, modifier = Modifier.weight(0.6f)) }
