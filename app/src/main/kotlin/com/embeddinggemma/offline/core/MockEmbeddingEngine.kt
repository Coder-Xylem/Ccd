package com.embeddinggemma.offline.core

import kotlin.math.abs

/** Deterministic hashed bag-of-words embedder. For UI work, unit tests and CI only. Never used when a real model is installed. */
class MockEmbeddingEngine(private val dims: Int = 768) : EmbeddingEngine {
    override val info = EngineInfo("Mock (tests only)", dims, text = true, image = true, audio = false, isMock = true)

    override fun embedText(formatted: String): FloatArray {
        val v = FloatArray(dims)
        val body = formatted.substringAfter("query: ", formatted).substringAfter("text: ", formatted)
        for (w in body.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.length > 1 }) {
            v[abs(w.hashCode()) % dims] += 1f
            if (w.length > 4) v[abs(w.take(4).hashCode() * 31) % dims] += 0.5f
        }
        return VectorMath.l2Normalize(v)
    }

    override fun embedImage(encoded: ByteArray): FloatArray {
        val v = FloatArray(dims)
        for (i in encoded.indices step 7) v[(encoded[i].toInt() and 0xff) * 3 % dims] += 1f
        return VectorMath.l2Normalize(v)
    }

    override fun embedTextAndImage(formatted: String, encoded: ByteArray) =
        VectorMath.l2Normalize(embedText(formatted).zip(embedImage(encoded)) { a, b -> a + b }.toFloatArray())

    override fun close() {}
}
