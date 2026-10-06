package com.embeddinggemma.offline.core

import java.util.PriorityQueue

data class IndexEntry(
    val chunkId: Long,
    val itemId: Long,
    val modality: Modality,
    val collection: String,
    val vector: FloatArray,
)

data class Hit(val entry: IndexEntry, val score: Float)

/** In-memory exact (brute-force) cosine index over unit-length vectors. Fine for ~100k vectors at 256 dims on a phone. */
class VectorIndex(val dims: Int) {
    private val entries = ArrayList<IndexEntry>()
    val size get() = entries.size

    @Synchronized fun add(e: IndexEntry) { require(e.vector.size == dims); entries.add(e) }
    @Synchronized fun removeItem(itemId: Long) { entries.removeAll { it.itemId == itemId } }
    @Synchronized fun clear() = entries.clear()
    @Synchronized fun snapshot(): List<IndexEntry> = entries.toList()

    @Synchronized
    fun search(query: FloatArray, k: Int, minScore: Float = -1f, filter: (IndexEntry) -> Boolean = { true }): List<Hit> {
        require(query.size == dims) { "query has ${query.size} dims, index has $dims" }
        val heap = PriorityQueue<Hit>(compareBy { it.score })
        for (e in entries) {
            if (!filter(e)) continue
            val s = VectorMath.dot(query, e.vector)
            if (s < minScore) continue
            if (heap.size < k) heap.add(Hit(e, s)) else if (s > heap.peek()!!.score) { heap.poll(); heap.add(Hit(e, s)) }
        }
        return heap.sortedByDescending { it.score }
    }
}
