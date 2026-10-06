package com.embeddinggemma.offline.data

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.embeddinggemma.offline.core.Chunker
import com.embeddinggemma.offline.core.Embedder
import com.embeddinggemma.offline.core.IndexEntry
import com.embeddinggemma.offline.core.Modality
import com.embeddinggemma.offline.core.VectorIndex
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

data class Progress(val done: Int = 0, val total: Int = 0, val current: String? = null, val paused: Boolean = false, val running: Boolean = false)

enum class VideoMode(val label: String, val frames: Int) { FAST("Fast", 4), BALANCED("Balanced", 8), DETAILED("Detailed", 16) }

object FileKinds {
    private val code = setOf("py", "cpp", "cc", "c", "h", "hpp", "java", "kt", "kts", "js", "ts", "tsx", "jsx", "html", "css", "sql", "go", "rs", "sh", "json", "xml", "yaml", "yml")
    private val text = setOf("txt", "md", "markdown", "rst", "csv", "log")
    fun modality(name: String, mime: String?): Modality? {
        val ext = name.substringAfterLast('.', "").lowercase()
        return when {
            mime?.startsWith("image/") == true -> Modality.IMAGE
            mime?.startsWith("video/") == true -> Modality.VIDEO
            ext in code -> Modality.CODE
            ext in text || mime?.startsWith("text/") == true -> Modality.TEXT
            else -> null // audio, PDF, etc. are not supported in this build
        }
    }
}

/** Persistent queue worker. The queue is the `items` table, so it survives process death; PROCESSING rows are re-queued on start. */
class Indexer(private val ctx: Context, private val store: Store, private val scope: CoroutineScope,
              private val embedder: () -> Embedder?, private val index: () -> VectorIndex, private val videoMode: () -> VideoMode,
              private val onChange: () -> Unit) {
    val progress = MutableStateFlow(Progress())
    @Volatile private var paused = false
    private var job: Job? = null

    fun kick() {
        paused = false
        if (job?.isActive == true) { publish(); return }
        job = scope.launch { loop() }
    }
    fun pause() { paused = true; publish() }
    /** Removes everything still waiting; finished items stay. */
    fun cancelQueued() { paused = true; store.items().filter { it.state == State.QUEUED }.forEach { store.deleteItem(it.id) }; publish(); onChange() }

    private fun publish(current: String? = null) {
        val (done, total) = store.queueStats()
        progress.value = Progress(done, total, current, paused, job?.isActive == true)
    }

    private suspend fun loop() {
        while (scope.isActive && !paused) {
            val em = embedder() ?: break
            val item = store.nextQueued() ?: break
            store.setState(item.id, State.PROCESSING); publish(item.name)
            try {
                val entries = process(item, em)
                entries.forEach { index().add(it) }
                store.setState(item.id, State.DONE)
            } catch (e: kotlinx.coroutines.CancellationException) {
                store.setState(item.id, State.QUEUED); throw e
            } catch (e: Throwable) {
                store.writableDatabase.execSQL("DELETE FROM chunks WHERE item_id=?", arrayOf(item.id)); store.setState(item.id, State.FAILED, e.message ?: e.javaClass.simpleName)
            }
            publish(); onChange()
        }
        publish(); onChange()
    }

    private fun readLimited(uri: String, max: Int): ByteArray {
        val bos = ByteArrayOutputStream(); val buf = ByteArray(64 * 1024)
        ctx.contentResolver.openInputStream(Uri.parse(uri))!!.use { i ->
            while (bos.size() < max) { val n = i.read(buf); if (n < 0) break; bos.write(buf, 0, n) }
        }
        return bos.toByteArray()
    }

    private fun readText(uri: String): String =
        readLimited(uri, 2_000_000).toString(Charsets.UTF_8)

    private fun process(item: Item, em: Embedder): List<IndexEntry> {
        val out = ArrayList<IndexEntry>()
        fun add(ord: Int, label: String?, text: String?, ls: Int, le: Int, v: FloatArray) {
            val id = store.addChunk(item.id, ord, label, text, ls, le, v)
            out += IndexEntry(id, item.id, item.modality, item.collection, v)
        }
        when (item.modality) {
            Modality.NOTE, Modality.TEXT -> {
                val raw = item.body ?: readText(item.uri)
                for (c in Chunker.text(raw)) add(c.ordinal, c.section, c.text, 0, 0, em.document(c.text, item.name + (c.section?.let { " / $it" } ?: "")))
            }
            Modality.CODE -> for (c in Chunker.code(readText(item.uri), item.name))
                add(c.ordinal, c.section, c.text, c.startLine, c.endLine, em.document(c.text, c.title))
            Modality.IMAGE -> {
                check(em.info.image) { "Installed model has no vision encoder (use a text+vision or omnimodal bundle)" }
                val bytes = readLimited(item.uri, 16_000_000)
                add(0, item.name, null, 0, 0, em.image(bytes))
            }
            Modality.VIDEO -> {
                check(em.info.image) { "Installed model has no vision encoder" }
                val r = MediaMetadataRetriever()
                try {
                    r.setDataSource(ctx, Uri.parse(item.uri))
                    val dur = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                    val n = videoMode().frames
                    for (i in 0 until n) {
                        val tMs = dur * (2 * i + 1) / (2 * n)
                        val bmp = r.getFrameAtTime(tMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: continue
                        val bos = ByteArrayOutputStream(); bmp.compress(Bitmap.CompressFormat.JPEG, 85, bos)
                        add(i, "%d:%02d".format(tMs / 60000, tMs / 1000 % 60), null, 0, 0, em.image(bos.toByteArray()))
                    }
                } finally { r.release() }
                check(out.isNotEmpty()) { "No frames could be decoded" }
            }
            Modality.AUDIO -> error("Audio embedding is not enabled in this build")
        }
        return out
    }
}
