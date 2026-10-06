package com.embeddinggemma.offline.core

data class EngineInfo(
    val name: String,
    val nativeDims: Int,
    val text: Boolean,
    val image: Boolean,
    val audio: Boolean,
    val isMock: Boolean,
)

/** Raw inference. Inputs are already task-formatted. Output is the model's native vector. Blocking: call off the main thread. */
interface EmbeddingEngine : AutoCloseable {
    val info: EngineInfo
    fun embedText(formatted: String): FloatArray
    /** [encoded] is a compressed image (JPEG/PNG/WebP). */
    fun embedImage(encoded: ByteArray): FloatArray
    /** Interleaved text + image content embedded as one input by the model. */
    fun embedTextAndImage(formatted: String, encoded: ByteArray): FloatArray
}

/**
 * Applies task formatting, Matryoshka truncation and re-normalisation on top of an engine,
 * so every stored vector and every query share the same dimensionality.
 */
class Embedder(private val engine: EmbeddingEngine, val dims: Int) {
    val info get() = engine.info
    init { require(dims in 1..engine.info.nativeDims) }

    private fun fit(v: FloatArray) = VectorMath.truncate(v, dims)

    fun query(text: String, task: EmbeddingTask = EmbeddingTask.SEARCH) = fit(engine.embedText(task.formatQuery(text)))
    fun document(text: String, title: String? = null, task: EmbeddingTask = EmbeddingTask.SEARCH) =
        fit(engine.embedText(task.formatDocument(text, title)))
    fun image(encoded: ByteArray) = fit(engine.embedImage(encoded))
    fun textAndImage(text: String, encoded: ByteArray, task: EmbeddingTask = EmbeddingTask.SEARCH) =
        fit(engine.embedTextAndImage(task.formatDocument(text), encoded))
}
