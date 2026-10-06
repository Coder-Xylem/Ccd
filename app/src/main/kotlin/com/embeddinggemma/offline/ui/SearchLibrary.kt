package com.embeddinggemma.offline.ui

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.embeddinggemma.offline.AppState
import com.embeddinggemma.offline.Thumb
import com.embeddinggemma.offline.core.EmbeddingTask
import com.embeddinggemma.offline.core.Modality
import com.embeddinggemma.offline.data.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val filters = listOf("All" to emptySet<Modality>(), "Text" to setOf(Modality.TEXT), "Images" to setOf(Modality.IMAGE),
    "Video" to setOf(Modality.VIDEO), "Code" to setOf(Modality.CODE), "Notes" to setOf(Modality.NOTE))

@Composable
fun ModelBanner(app: AppState) {
    val eng by app.engine.collectAsState(); val st by app.modelStatus.collectAsState()
    if (eng == null) Text("Model: $st. Import EmbeddingGemma 2 in Settings to index and search.", color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(vertical = 8.dp))
}

@Composable
fun SearchScreen(app: AppState) {
    val ctx = LocalContext.current; val scope = rememberCoroutineScope()
    val version by app.version.collectAsState(); val eng by app.engine.collectAsState(); val dims by app.dims.collectAsState()
    var q by remember { mutableStateOf("") }
    var filter by remember { mutableIntStateOf(0) }
    var ask by remember { mutableStateOf(false) }
    var hybrid by remember { mutableStateOf(true) }
    var collection by remember { mutableStateOf<String?>(null) }
    var minScore by remember { mutableFloatStateOf(0f) }
    var task by remember { mutableStateOf<EmbeddingTask?>(null) }
    var advanced by remember { mutableStateOf(false) }
    var results by remember { mutableStateOf<List<Result>?>(null) }
    var note by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var job by remember { mutableStateOf<Job?>(null) }
    val history = remember(version, results) { app.store.history() }

    fun run(block: suspend () -> List<Result>) {
        job?.cancel(); busy = true; note = null
        job = scope.launch {
            try { results = withContext(Dispatchers.Default) { block() } }
            catch (e: kotlinx.coroutines.CancellationException) { throw e }
            catch (t: Throwable) { note = t.message ?: "Search failed" }
            busy = false
        }
    }
    fun filterObj(): SearchFilter {
        val mods = if (ask) setOf(Modality.TEXT, Modality.NOTE, Modality.CODE) else filters[filter].second
        return SearchFilter(mods, collection, minScore)
    }
    fun doSearch(text: String = q) {
        if (text.isBlank()) return
        val em = app.embedder() ?: run { note = "Install the model first (Settings)."; return }
        app.store.addHistory(text)
        val t = task ?: when { ask -> EmbeddingTask.QUESTION_ANSWERING; filter == 4 -> EmbeddingTask.CODE_RETRIEVAL; else -> EmbeddingTask.SEARCH }
        run { app.search.search(em.query(text, t), text, filterObj(), hybrid) }
    }
    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        val em = app.embedder() ?: run { note = "Install the model first (Settings)."; return@rememberLauncherForActivityResult }
        if (!em.info.image) { note = "Installed model has no vision encoder."; return@rememberLauncherForActivityResult }
        run {
            val bytes = ctx.contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            app.search.search(em.image(bytes), null, filterObj(), false)
        }
    }

    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        Text("Search everything", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(top = 16.dp, bottom = 8.dp))
        ModelBanner(app)
        OutlinedTextField(q, { q = it }, Modifier.fillMaxWidth(), singleLine = true, placeholder = { Text(if (ask) "Ask a question about your files" else "Describe what you are looking for") },
            keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(imeAction = ImeAction.Search), keyboardActions = KeyboardActions(onSearch = { doSearch() }))
        Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { doSearch() }, enabled = eng != null && !busy) { Text(if (ask) "Find passages" else "Search") }
            OutlinedButton(onClick = { pickImage.launch("image/*") }, enabled = eng?.info?.image == true) { Text("Image query") }
            TextButton(onClick = { advanced = !advanced }) { Text(if (advanced) "Less" else "More") }
        }
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            filters.forEachIndexed { i, f -> FilterChip(filter == i, { filter = i }, { Text(f.first) }) }
        }
        if (advanced) Column(Modifier.padding(top = 8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) { Switch(ask, { ask = it }); Spacer(Modifier.width(8.dp)); Text("Ask my files (passages only)") }
            Row(verticalAlignment = Alignment.CenterVertically) { Switch(hybrid, { hybrid = it }); Spacer(Modifier.width(8.dp)); Text("Hybrid keyword boost") }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Chooser("Collection", listOf<String?>(null) + app.store.collections(), collection, { it ?: "All" }) { collection = it }
                Chooser("Task", listOf<EmbeddingTask?>(null) + allTasks, task, { it?.label ?: "Auto" }) { task = it }
            }
            Text("Min similarity ${pct(minScore)}"); Slider(minScore, { minScore = it }, valueRange = 0f..0.8f)
            Muted("Indexed vectors: ${app.index.size} at $dims dimensions")
        }
        note?.let { Text(it, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(vertical = 8.dp)) }
        if (busy) LinearProgressIndicator(Modifier.fillMaxWidth().padding(vertical = 8.dp))
        val r = results
        LazyColumn(Modifier.fillMaxSize()) {
            if (r == null) {
                if (history.isNotEmpty()) {
                    item { Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Title("Recent"); TextButton(onClick = { app.store.clearHistory(); app.version.value++ }) { Text("Clear") } } }
                    items(history) { h -> Text(h, Modifier.fillMaxWidth().clickable { q = h; doSearch(h) }.padding(vertical = 10.dp)) }
                }
            } else {
                item { Title(if (r.isEmpty()) "No results" else if (ask) "Most relevant passages" else "Results") }
                items(r, key = { it.chunk.id }) { res -> ResultRow(app, res) { vec -> run { app.search.search(vec, null, SearchFilter(), false) } } }
            }
        }
    }
}

@Composable
private fun ResultRow(app: AppState, r: Result, similar: (FloatArray) -> Unit) {
    Column(Modifier.fillMaxWidth().padding(vertical = 10.dp)) {
        Row(verticalAlignment = Alignment.Top) {
            if (r.item.modality == Modality.IMAGE) { Thumb(r.item.uri); Spacer(Modifier.width(12.dp)) }
            Column(Modifier.weight(1f)) {
                Text(r.item.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                val where = buildList {
                    add(r.item.modality.nice()); r.chunk.label?.let { add(it) }
                    if (r.chunk.lineEnd > 0) add("lines ${r.chunk.lineStart}–${r.chunk.lineEnd}")
                    add(r.item.collection)
                }.joinToString(" · ")
                Muted(where)
            }
            Text(pct(r.score), style = MaterialTheme.typography.titleSmall)
        }
        r.chunk.text?.let { Text(it, maxLines = 4, overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp)) }
        TextButton(onClick = { app.store.vector(r.chunk.id)?.let(similar) }) { Text("Find similar") }
        Rule()
    }
}

@Composable
fun LibraryScreen(app: AppState) {
    val ctx = LocalContext.current
    val version by app.version.collectAsState(); val prog by app.indexer.progress.collectAsState(); val eng by app.engine.collectAsState()
    var collection by remember { mutableStateOf("Inbox") }
    var newColl by remember { mutableStateOf(false) }; var newNote by remember { mutableStateOf(false) }
    val counts = remember(version) { app.store.counts() }
    val items = remember(version, collection) { app.store.items(collection) }
    val colls = remember(version) { app.store.collections() }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        uris.forEach { runCatching { ctx.contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION) } }
        app.addFiles(uris, collection)
    }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        item { Text("My Knowledge", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(top = 16.dp)); ModelBanner(app) }
        item {
            val rows = listOf("Documents" to Modality.TEXT, "Images" to Modality.IMAGE, "Video" to Modality.VIDEO, "Notes" to Modality.NOTE, "Code" to Modality.CODE)
            rows.forEach { (n, m) -> Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), horizontalArrangement = Arrangement.SpaceBetween) { Text(n); Text("${counts[m.name] ?: 0}") } }
            Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), horizontalArrangement = Arrangement.SpaceBetween) { Text("Indexed"); Text("${prog.done}") }
            Rule()
        }
        item {
            Row(Modifier.horizontalScroll(rememberScrollState()).padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                colls.forEach { c -> FilterChip(collection == c, { collection = c }, { Text(c) }) }
                AssistChip({ newColl = true }, { Text("+ Collection") })
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(vertical = 8.dp)) {
                Button(onClick = { pick.launch(arrayOf("*/*")) }) { Text("Add files") }
                OutlinedButton(onClick = { newNote = true }) { Text("New note") }
            }
            Muted("Supported: text, Markdown, source code, images, video (frame sampling). PDF and audio are not supported in this build.")
        }
        if (prog.total > 0 && prog.done < prog.total) item {
            Column(Modifier.padding(vertical = 8.dp)) {
                Text("Indexing…  ${prog.done} / ${prog.total}  ·  ${prog.done * 100 / prog.total}%")
                prog.current?.let { Muted(it) }
                LinearProgressIndicator(progress = { prog.done.toFloat() / prog.total }, modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (prog.paused || !prog.running) OutlinedButton(onClick = { app.indexer.kick() }, enabled = eng != null) { Text("Resume") }
                    else OutlinedButton(onClick = { app.indexer.pause() }) { Text("Pause") }
                    OutlinedButton(onClick = { app.indexer.cancelQueued() }) { Text("Cancel waiting") }
                    if (items.any { it.state == State.FAILED }) OutlinedButton(onClick = { app.store.requeueFailed(); app.indexer.kick() }) { Text("Retry failed") }
                }
            }
        }
        item { Title("Index queue · $collection") }
        items(items, key = { it.id }) { it ->
            Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(it.name, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Muted("${it.modality.nice()} · ${when (it.state) { State.QUEUED -> "Waiting"; State.PROCESSING -> "Processing"; State.DONE -> "Complete"; State.FAILED -> "Failed: ${it.error}" }}")
                }
                TextButton(onClick = { app.deleteItem(it.id) }) { Text("Remove") }
            }
        }
        item { Spacer(Modifier.height(24.dp)) }
    }
    if (newColl) { var n by remember { mutableStateOf("") }
        AlertDialog({ newColl = false }, title = { Text("New collection") }, text = { OutlinedTextField(n, { n = it }, singleLine = true) },
            confirmButton = { TextButton({ if (n.isNotBlank()) { app.store.addCollection(n); collection = n.trim(); app.version.value++ }; newColl = false }) { Text("Create") } },
            dismissButton = { TextButton({ newColl = false }) { Text("Cancel") } }) }
    if (newNote) { var t by remember { mutableStateOf("") }; var b by remember { mutableStateOf("") }
        AlertDialog({ newNote = false }, title = { Text("New note") }, text = { Column { OutlinedTextField(t, { t = it }, singleLine = true, placeholder = { Text("Title") }); OutlinedTextField(b, { b = it }, minLines = 4, placeholder = { Text("Write…") }) } },
            confirmButton = { TextButton({ if (b.isNotBlank()) app.addNote(t, b, collection); newNote = false }) { Text("Save") } },
            dismissButton = { TextButton({ newNote = false }) { Text("Cancel") } }) }
}
