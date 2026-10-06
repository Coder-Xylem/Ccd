package com.embeddinggemma.offline.data

import com.embeddinggemma.offline.core.Hybrid
import com.embeddinggemma.offline.core.Modality
import com.embeddinggemma.offline.core.VectorIndex

data class SearchFilter(
    val modalities: Set<Modality> = emptySet(),
    val collection: String? = null,
    val minScore: Float = 0f,
    val sinceMs: Long = 0L,
    val extension: String? = null,
)

data class Result(val item: Item, val chunk: Chunk, val score: Float)

/** query vector -> vector search -> filter -> (optional keyword fusion) -> ranked results. Independent of the UI. */
class SemanticSearchEngine(private val store: Store, private val index: () -> VectorIndex) {
    fun search(query: FloatArray, queryText: String?, f: SearchFilter, hybrid: Boolean, k: Int = 30): List<Result> {
        val idx = index()
        if (query.size != idx.dims) return emptyList()
        val hits = idx.search(query, k * 3, f.minScore) {
            (f.modalities.isEmpty() || it.modality in f.modalities) && (f.collection == null || it.collection == f.collection)
        }
        var out = hits.mapNotNull { h ->
            val c = store.chunk(h.entry.chunkId) ?: return@mapNotNull null
            val i = store.item(h.entry.itemId) ?: return@mapNotNull null
            if (i.addedAt < f.sinceMs) return@mapNotNull null
            if (f.extension != null && !i.name.endsWith(f.extension, true)) return@mapNotNull null
            Result(i, c, h.score)
        }
        if (hybrid && !queryText.isNullOrBlank())
            out = out.map { it.copy(score = Hybrid.fuse(it.score, Hybrid.keywordScore(queryText, (it.chunk.text ?: "") + " " + it.item.name))) }
        return out.sortedByDescending { it.score }.take(k)
    }
}
