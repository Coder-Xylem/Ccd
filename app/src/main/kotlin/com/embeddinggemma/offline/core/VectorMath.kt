package com.embeddinggemma.offline.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

object VectorMath {
    val DIMENSIONS = listOf(768, 512, 256, 128)

    fun norm(v: FloatArray): Float { var s = 0.0; for (x in v) s += x * x; return sqrt(s).toFloat() }

    fun l2Normalize(v: FloatArray): FloatArray {
        val n = norm(v)
        return if (n == 0f) v.copyOf() else FloatArray(v.size) { v[it] / n }
    }

    /** Matryoshka: keep the first [dims] values, then re-normalise (slicing alone breaks unit length). */
    fun truncate(v: FloatArray, dims: Int): FloatArray {
        require(dims in 1..v.size) { "dims $dims out of range 1..${v.size}" }
        return l2Normalize(v.copyOf(dims))
    }

    fun dot(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "dimension mismatch ${a.size} vs ${b.size}" }
        var s = 0f; for (i in a.indices) s += a[i] * b[i]; return s
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        val na = norm(a); val nb = norm(b)
        return if (na == 0f || nb == 0f) 0f else dot(a, b) / (na * nb)
    }

    fun mean(vs: List<FloatArray>): FloatArray {
        val out = FloatArray(vs.first().size)
        for (v in vs) for (i in out.indices) out[i] += v[i]
        return l2Normalize(FloatArray(out.size) { out[it] / vs.size })
    }

    fun toBytes(v: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(v); return bb.array()
    }

    fun fromBytes(b: ByteArray): FloatArray {
        val fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }
}
