package com.embeddinggemma.offline

import android.app.Application
import android.content.Context
import android.net.Uri
import com.embeddinggemma.offline.core.Embedder
import com.embeddinggemma.offline.core.EmbeddingEngine
import com.embeddinggemma.offline.core.VectorIndex
import com.embeddinggemma.offline.data.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import java.io.File

class OfflineApp : Application() {
    lateinit var state: AppState
    override fun onCreate() { super.onCreate(); state = AppState(this) }
}

/** Process-wide state. All durable data lives in SQLite / SharedPreferences, so process death only loses in-memory caches. */
class AppState(private val ctx: Context) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val store = Store(ctx)
    private val prefs = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)
    val engine = MutableStateFlow<EmbeddingEngine?>(null)
    val modelStatus = MutableStateFlow("No model installed")
    val version = MutableStateFlow(0) // bumped whenever stored data changes, so screens refresh
    val dims = MutableStateFlow(prefs.getInt("dims", 512))
    val videoMode = MutableStateFlow(VideoMode.valueOf(prefs.getString("video", "BALANCED")!!))
    @Volatile var index = VectorIndex(dims.value); private set
    val search = SemanticSearchEngine(store) { index }
    val modelDir = File(ctx.filesDir, "models").apply { mkdirs() }

    val indexer = Indexer(ctx, store, scope, { embedder() }, { index }, { videoMode.value }, { version.value++ })

    fun embedder(): Embedder? = engine.value?.let { Embedder(it, dims.value) }
    fun modelFile(): File? = modelDir.listFiles()?.firstOrNull { it.name.endsWith(".litertlm") }
    fun variant(): ModelVariant? = modelFile()?.let { ModelVariant.detect(it.name) }

    init {
        scope.launch {
            store.recoverInterrupted(); reloadIndex(); loadModel()
            if (store.nextQueued() != null && engine.value != null) indexer.kick()
        }
    }

    fun reloadIndex() { val i = VectorIndex(dims.value); store.loadEntries(dims.value).forEach { i.add(it) }; index = i; version.value++ }

    fun loadModel() {
        engine.value?.close(); engine.value = null
        val f = modelFile() ?: run { modelStatus.value = "No model installed"; return }
        modelStatus.value = "Loading ${f.name}…"
        try {
            engine.value = EmbeddingGemma2Engine(ctx, f, ModelVariant.detect(f.name))
            modelStatus.value = "Ready"
        } catch (t: Throwable) { modelStatus.value = "Failed to load: ${t.message}" }
    }

    fun importModel(uri: Uri) = scope.launch {
        modelStatus.value = "Copying model…"
        val name = (ctx.contentResolver.query(uri, arrayOf("_display_name"), null, null, null)?.use { if (it.moveToFirst()) it.getString(0) else null } ?: "model.litertlm")
            .let { if (it.endsWith(".litertlm")) it else "$it.litertlm" }
        modelDir.listFiles()?.forEach { it.delete() }
        try {
            ctx.contentResolver.openInputStream(uri)!!.use { i -> File(modelDir, name).outputStream().use { o -> i.copyTo(o, 1 shl 20) } }
            loadModel(); if (engine.value != null) indexer.kick()
        } catch (t: Throwable) { modelStatus.value = "Import failed: ${t.message}" }
    }

    fun removeModel() { indexer.pause(); engine.value?.close(); engine.value = null; modelDir.listFiles()?.forEach { it.delete() }; modelStatus.value = "No model installed" }

    /** Changing dimensionality invalidates stored vectors: they are dropped and everything is re-queued. */
    fun setDims(d: Int) = scope.launch {
        if (d == dims.value) return@launch
        indexer.pause(); prefs.edit().putInt("dims", d).apply(); dims.value = d
        store.clearEmbeddings(); index = VectorIndex(d); version.value++
        if (engine.value != null) indexer.kick()
    }
    fun setVideoMode(m: VideoMode) { videoMode.value = m; prefs.edit().putString("video", m.name).apply() }

    fun addFiles(uris: List<Uri>, collection: String) = scope.launch {
        for (u in uris) {
            val (name, mime) = ctx.contentResolver.query(u, arrayOf("_display_name"), null, null, null)?.use { if (it.moveToFirst()) it.getString(0) else null }.let { (it ?: u.lastPathSegment ?: "file") to ctx.contentResolver.getType(u) }
            val m = FileKinds.modality(name, mime) ?: continue
            store.addItem(u.toString(), name, m, collection)
        }
        version.value++; indexer.kick()
    }
    fun addNote(title: String, body: String, collection: String) = scope.launch {
        store.addItem("note:${System.nanoTime()}", title.ifBlank { body.take(40) }, com.embeddinggemma.offline.core.Modality.NOTE, collection, body)
        version.value++; indexer.kick()
    }
    fun deleteItem(id: Long) { store.deleteItem(id); index.removeItem(id); version.value++ }

    fun deleteEmbeddings() { indexer.pause(); store.clearEmbeddings(); index = VectorIndex(dims.value); version.value++ }
    fun deleteIndexedData() { indexer.pause(); store.clearIndexedData(); index = VectorIndex(dims.value); version.value++ }
    fun deleteAllAppData() { deleteIndexedData(); store.clearAll(); removeModel(); prefs.edit().clear().apply(); dims.value = 512; videoMode.value = VideoMode.BALANCED }
}
