package com.embeddinggemma.offline.core

import org.junit.Assert.*
import org.junit.Test

class CoreTest {
    @Test fun taskFormatting() {
        assertEquals("task: search result | query: hi", EmbeddingTask.SEARCH.formatQuery("hi"))
        assertEquals("title: none | text: body", EmbeddingTask.SEARCH.formatDocument("body"))
        assertEquals("title: f.kt | text: x", EmbeddingTask.CODE_RETRIEVAL.formatDocument("x", "f.kt"))
        assertEquals("task: clustering | query: a", EmbeddingTask.CLUSTERING.formatDocument("a"))
        assertEquals("task: question answering | query: q?", EmbeddingTask.QUESTION_ANSWERING.formatQuery("q?"))
    }
    @Test fun normalizeAndTruncate() {
        val v = FloatArray(768) { (it % 7 + 1).toFloat() }
        assertEquals(1f, VectorMath.norm(VectorMath.l2Normalize(v)), 1e-5f)
        for (d in VectorMath.DIMENSIONS) {
            val t = VectorMath.truncate(v, d)
            assertEquals(d, t.size); assertEquals(1f, VectorMath.norm(t), 1e-5f)
        }
        val raw = v.copyOf(128); assertTrue(VectorMath.norm(raw) > 1.5f) // slicing alone would not be unit length
    }
    @Test fun cosineAndBytes() {
        val a = floatArrayOf(1f, 0f); val b = floatArrayOf(0f, 1f)
        assertEquals(0f, VectorMath.cosine(a, b), 1e-6f); assertEquals(1f, VectorMath.cosine(a, a), 1e-6f)
        assertArrayEquals(a, VectorMath.fromBytes(VectorMath.toBytes(a)), 0f)
    }
    @Test fun searchFilterRanking() {
        val e = MockEmbeddingEngine(); val em = Embedder(e, 256); val idx = VectorIndex(256)
        val docs = listOf("grain refinement strengthens metals", "chocolate cake recipe with eggs", "precipitation hardening of alloys")
        docs.forEachIndexed { i, d -> idx.add(IndexEntry(i.toLong(), i.toLong(), if (i == 2) Modality.NOTE else Modality.TEXT, "all", em.document(d))) }
        val hits = idx.search(em.query("metals grain refinement"), 3)
        assertEquals(0L, hits.first().entry.chunkId); assertTrue(hits[0].score >= hits[1].score)
        val f = idx.search(em.query("alloys hardening"), 3) { it.modality == Modality.NOTE }
        assertEquals(1, f.size); assertEquals(2L, f[0].entry.chunkId)
    }
    @Test(expected = IllegalArgumentException::class) fun dimMismatchRejected() {
        VectorIndex(128).search(FloatArray(256), 1)
    }
    @Test fun chunking() {
        val md = "# Intro\n\nHello world.\n\n## Methods\n\n" + "Sentence one. ".repeat(200)
        val c = Chunker.text(md, 500)
        assertTrue(c.size > 2); assertTrue(c.all { it.text.length <= 520 }); assertEquals("Intro", c.first().section)
        assertTrue(c.any { it.section == "Methods" })
        val code = "import x\n\nfun a() {\n  1\n}\n\nclass B {\n}\n"
        val cc = Chunker.code(code, "T.kt")
        assertTrue(cc.any { it.section == "a" }); assertTrue(cc.any { it.section == "B" && it.startLine == 7 })
    }
    @Test fun clusteringSeparates() {
        val t = listOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(1f, 0.1f, 0f),
            floatArrayOf(0f, 1f, 0f),
            floatArrayOf(0f, 1f, 0.1f)
        ).map { VectorMath.l2Normalize(it) }
        val a = KMeans.cluster(t, 2)
        assertEquals(a[0], a[1]); assertEquals(a[2], a[3]); assertNotEquals(a[0], a[2])
    }
    @Test fun duplicates() {
        val em = Embedder(MockEmbeddingEngine(), 256)
        val m = mapOf(1L to em.document("research notes on zif-8 membranes"), 2L to em.document("research notes on zif-8 membranes"), 3L to em.document("banana bread"))
        val d = Duplicates.find(m, 0.95f); assertEquals(1, d.size); assertEquals(setOf(1L, 2L), setOf(d[0].a, d[0].b))
    }
    @Test fun zeroShotAndHybrid() {
        val em = Embedder(MockEmbeddingEngine(), 256)
        val labels = listOf("Research", "Finance", "Cooking").map { it to em.document("$it $it", task = EmbeddingTask.CLASSIFICATION) }
        val r = ZeroShot.rank(em.document("research project research", task = EmbeddingTask.CLASSIFICATION), labels)
        assertEquals("Research", r.first().label); assertEquals(1f, r.sumOf { it.share.toDouble() }.toFloat(), 1e-4f)
        assertTrue(Hybrid.keywordScore("jwt validation", "where is JWT validation") > 0.9f)
    }
}
