package com.embeddinggemma.offline.core

import kotlin.math.exp
import kotlin.random.Random

/** Cosine-ranked zero-shot labels. [share] is a softmax over the label scores, a relative measure, NOT a calibrated probability. */
data class LabelScore(val label: String, val similarity: Float, val share: Float)

object ZeroShot {
    fun rank(input: FloatArray, labels: List<Pair<String, FloatArray>>, temperature: Float = 0.05f): List<LabelScore> {
        val sims = labels.map { VectorMath.dot(input, it.second) }
        val mx = sims.max()
        val ex = sims.map { exp(((it - mx) / temperature).toDouble()).toFloat() }
        val sum = ex.sum()
        return labels.indices.map { LabelScore(labels[it].first, sims[it], ex[it] / sum) }.sortedByDescending { it.similarity }
    }
}

/** Spherical k-means (k-means++ init, fixed seed, deterministic). Returns a cluster index per input vector. */
object KMeans {
    fun suggestK(n: Int) = if (n < 4) 1 else minOf(12, maxOf(2, Math.round(Math.sqrt(n / 2.0)).toInt()))

    fun cluster(vs: List<FloatArray>, k: Int, iters: Int = 30, seed: Int = 7): IntArray {
        if (vs.isEmpty()) return IntArray(0)
        val kk = k.coerceIn(1, vs.size); val rnd = Random(seed)
        val cents = ArrayList<FloatArray>().apply { add(vs[rnd.nextInt(vs.size)].copyOf()) }
        while (cents.size < kk) {
            val d = DoubleArray(vs.size) { i -> 1.0 - cents.maxOf { VectorMath.dot(vs[i], it) }.coerceIn(-1f, 1f) }
            var r = rnd.nextDouble() * d.sum(); var pick = vs.size - 1
            for (i in d.indices) { r -= d[i]; if (r <= 0) { pick = i; break } }
            cents.add(vs[pick].copyOf())
        }
        val asg = IntArray(vs.size) { -1 }
        repeat(iters) {
            var changed = false
            for (i in vs.indices) {
                var best = 0; var bs = -2f
                for (c in cents.indices) { val s = VectorMath.dot(vs[i], cents[c]); if (s > bs) { bs = s; best = c } }
                if (asg[i] != best) { asg[i] = best; changed = true }
            }
            if (!changed) return asg
            for (c in cents.indices) {
                val m = vs.indices.filter { asg[it] == c }
                if (m.isNotEmpty()) cents[c] = VectorMath.mean(m.map { vs[it] })
            }
        }
        return asg
    }
}

data class DupPair(val a: Long, val b: Long, val similarity: Float)

object Duplicates {
    /** Pairs of items whose (unit-length) vectors have cosine >= [threshold], most similar first. Never deletes anything. */
    fun find(items: Map<Long, FloatArray>, threshold: Float = 0.92f): List<DupPair> {
        val ids = items.keys.toList(); val out = ArrayList<DupPair>()
        for (i in ids.indices) for (j in i + 1 until ids.size) {
            val s = VectorMath.dot(items.getValue(ids[i]), items.getValue(ids[j]))
            if (s >= threshold) out.add(DupPair(ids[i], ids[j], s))
        }
        return out.sortedByDescending { it.similarity }
    }
}

object Hybrid {
    private val tok = Regex("[^\\p{L}\\p{N}_.-]+")
    fun keywordScore(query: String, text: String): Float {
        val q = query.lowercase().split(tok).filter { it.length >= 3 }.toSet()
        if (q.isEmpty()) return 0f
        val t = text.lowercase()
        return q.count { t.contains(it) }.toFloat() / q.size
    }
    /** Weighted fusion; semantic score dominates, exact identifiers get a boost. */
    fun fuse(semantic: Float, keyword: Float, keywordWeight: Float = 0.15f) = (1 - keywordWeight) * semantic + keywordWeight * keyword
}
