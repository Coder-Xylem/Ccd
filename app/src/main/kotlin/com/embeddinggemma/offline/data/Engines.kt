package com.embeddinggemma.offline.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.embeddinggemma.offline.core.EmbeddingEngine
import com.embeddinggemma.offline.core.EngineInfo
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.retrieval.universalembedder.UniversalEmbedder
import com.google.mediapipe.tasks.retrieval.universalembedder.UniversalEmbedderOptions
import java.io.File

/** Which official LiteRT Community bundle a file is, judged by file name (text 270M / text+vision 440M / omnimodal 740M). */
enum class ModelVariant(val label: String, val params: String, val text: Boolean, val vision: Boolean, val audioEncoder: Boolean) {
    TEXT("Text", "270M", true, false, false),
    TEXT_VISION("Text + Vision", "440M", true, true, false),
    OMNIMODAL("Omnimodal (text, vision, audio)", "740M", true, true, true);

    companion object {
        fun detect(fileName: String): ModelVariant {
            val n = fileName.lowercase()
            return when { "text-vision" in n || "440m" in n -> TEXT_VISION; "740m" in n || "omni" in n -> OMNIMODAL; else -> TEXT }
        }
    }
}

/**
 * Real EmbeddingGemma 2 inference through MediaPipe Tasks' UniversalEmbedder (which wraps LiteRT-LM).
 * API per https://developers.google.com/edge/mediapipe/solutions/retrieval/universal_embedder/android
 * Audio input is not wired in this build, so [EngineInfo.audio] is always false.
 */
class EmbeddingGemma2Engine(context: Context, modelFile: File, val variant: ModelVariant) : EmbeddingEngine {
    private val embedder: UniversalEmbedder
    override val info = EngineInfo("EmbeddingGemma 2 ${variant.params}", 768, variant.text, variant.vision, audio = false, isMock = false)

    init {
        val b = UniversalEmbedderOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile.absolutePath).build())
            .setTextDelegate(Delegate.CPU)
            .setL2Normalize(true)
        if (variant.vision) b.setVisionDelegate(Delegate.CPU)
        embedder = UniversalEmbedder.createFromOptions(context, b.build())
    }

    override fun embedText(formatted: String): FloatArray = embedder.embedText(formatted).embeddings()[0].floatEmbedding()

    override fun embedImage(encoded: ByteArray): FloatArray {
        check(info.image) { "Installed model has no vision encoder" }
        return embedder.embedImage(BitmapImageBuilder(decode(encoded)).build()).embeddings()[0].floatEmbedding()
    }

    override fun embedTextAndImage(formatted: String, encoded: ByteArray): FloatArray {
        check(info.image) { "Installed model has no vision encoder" }
        val contents: List<Any> = listOf(formatted, BitmapImageBuilder(decode(encoded)).build())
        return embedder.embedContent(contents).embeddings()[0].floatEmbedding()
    }

    private fun decode(b: ByteArray, maxSide: Int = 1024): Bitmap {
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(b, 0, b.size, o)
        var s = 1; while (maxOf(o.outWidth, o.outHeight) / s > maxSide * 2) s *= 2
        val bm = BitmapFactory.decodeByteArray(b, 0, b.size, BitmapFactory.Options().apply { inSampleSize = s })
            ?: error("Could not decode image")
        return bm.copy(Bitmap.Config.ARGB_8888, false)
    }

    override fun close() = embedder.close()
}
