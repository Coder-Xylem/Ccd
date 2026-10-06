package com.embeddinggemma.offline.core

data class TextChunk(val text: String, val section: String?, val ordinal: Int, val startLine: Int = 0, val endLine: Int = 0, val title: String? = null)

object Chunker {
    /** Heading -> paragraph -> sentence aware chunking. Chunks never exceed [maxChars] unless a single word does. */
    fun text(raw: String, maxChars: Int = 1200): List<TextChunk> {
        val out = ArrayList<TextChunk>()
        var section: String? = null
        val buf = StringBuilder()
        fun flush() {
            val t = buf.toString().trim()
            if (t.isNotEmpty()) out.add(TextChunk(t, section, out.size))
            buf.clear()
        }
        val blocks = raw.replace("\r\n", "\n").split(Regex("\n\\s*\n"))
        for (block in blocks) {
            val b = block.trim()
            if (b.isEmpty()) continue
            val heading = Regex("^#{1,6}\\s+(.+)$").find(b.lineSequence().first())
            if (heading != null) { flush(); section = heading.groupValues[1].trim() }
            for (piece in splitLong(b, maxChars)) {
                if (buf.length + piece.length + 2 > maxChars) flush()
                if (buf.isNotEmpty()) buf.append("\n\n")
                buf.append(piece)
            }
        }
        flush()
        return out
    }

    private fun splitLong(p: String, max: Int): List<String> {
        if (p.length <= max) return listOf(p)
        val parts = ArrayList<String>(); val cur = StringBuilder()
        for (s in p.split(Regex("(?<=[.!?।])\\s+"))) {
            if (cur.length + s.length + 1 > max && cur.isNotEmpty()) { parts.add(cur.toString()); cur.clear() }
            if (s.length > max) { s.chunked(max).forEach { parts.add(it) } } else { if (cur.isNotEmpty()) cur.append(' '); cur.append(s) }
        }
        if (cur.isNotEmpty()) parts.add(cur.toString())
        return parts
    }

    private val decl = Regex("""^\s*((export\s+)?(public|private|protected|internal|static|abstract|open|data|suspend|async)?\s*)*(fun|def|class|object|interface|function|struct|void|int|bool|auto|CREATE\s+(TABLE|VIEW|FUNCTION))\b.*""", RegexOption.IGNORE_CASE)
    private val nameRe = Regex("""(?:fun|def|class|object|interface|function|struct|TABLE|VIEW)\s+([A-Za-z_][\w.<>]*)""", RegexOption.IGNORE_CASE)

    /** Splits source at top-level declarations (column-0 or lightly indented). Falls back to line windows. */
    fun code(raw: String, fileName: String, maxLines: Int = 80): List<TextChunk> {
        val lines = raw.replace("\r\n", "\n").lines()
        val starts = lines.indices.filter { i -> lines[i].isNotBlank() && (lines[i].length - lines[i].trimStart().length) <= 4 && decl.matches(lines[i]) }
        val bounds = if (starts.isEmpty()) lines.indices.step(maxLines).toList() else (if (starts.first() > 0) listOf(0) else emptyList()) + starts
        val out = ArrayList<TextChunk>()
        for ((n, s) in bounds.withIndex()) {
            val e = minOf(if (n + 1 < bounds.size) bounds[n + 1] else lines.size, s + maxLines)
            val body = lines.subList(s, e).joinToString("\n").trim()
            if (body.isEmpty()) continue
            val name = nameRe.find(lines[s])?.groupValues?.get(1)
            out.add(TextChunk(body, name, out.size, s + 1, e, title = if (name != null) "$fileName :: $name" else fileName))
        }
        return out
    }
}
