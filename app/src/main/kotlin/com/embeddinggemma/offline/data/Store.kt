package com.embeddinggemma.offline.data

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.embeddinggemma.offline.core.IndexEntry
import com.embeddinggemma.offline.core.Modality
import com.embeddinggemma.offline.core.VectorMath

enum class State { QUEUED, PROCESSING, DONE, FAILED }

data class Item(val id: Long, val uri: String, val name: String, val modality: Modality, val collection: String,
                val addedAt: Long, val state: State, val error: String?, val body: String?)

data class Chunk(val id: Long, val itemId: Long, val ord: Int, val label: String?, val text: String?, val lineStart: Int, val lineEnd: Int, val dims: Int)

/** Local SQLite store. Vectors are little-endian float32 BLOBs, one row per chunk/frame, stored at the dimensionality active at index time. */
class Store(ctx: Context) : SQLiteOpenHelper(ctx, "knowledge.db", null, 1) {
    override fun onConfigure(db: SQLiteDatabase) { db.setForeignKeyConstraintsEnabled(true) }
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT, uri TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
            modality TEXT NOT NULL, collection TEXT NOT NULL, added_at INTEGER NOT NULL, state TEXT NOT NULL, error TEXT, body TEXT)""")
        db.execSQL("""CREATE TABLE chunks(id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            ord INTEGER NOT NULL, label TEXT, text TEXT, line_start INTEGER NOT NULL DEFAULT 0, line_end INTEGER NOT NULL DEFAULT 0,
            dims INTEGER NOT NULL, vec BLOB NOT NULL)""")
        db.execSQL("CREATE INDEX chunks_item ON chunks(item_id)")
        db.execSQL("CREATE TABLE collections(name TEXT PRIMARY KEY)")
        db.execSQL("INSERT INTO collections VALUES('Inbox')")
        db.execSQL("CREATE TABLE history(id INTEGER PRIMARY KEY AUTOINCREMENT, q TEXT NOT NULL, ts INTEGER NOT NULL)")
    }
    override fun onUpgrade(db: SQLiteDatabase, o: Int, n: Int) {}

    private fun item(c: Cursor) = Item(c.getLong(0), c.getString(1), c.getString(2), Modality.valueOf(c.getString(3)), c.getString(4),
        c.getLong(5), State.valueOf(c.getString(6)), c.getString(7), c.getString(8))
    private val itemCols = "id,uri,name,modality,collection,added_at,state,error,body"

    @Synchronized fun addItem(uri: String, name: String, m: Modality, collection: String, body: String? = null): Long {
        val cv = ContentValues().apply { put("uri", uri); put("name", name); put("modality", m.name); put("collection", collection)
            put("added_at", System.currentTimeMillis()); put("state", State.QUEUED.name); put("body", body) }
        return writableDatabase.insertWithOnConflict("items", null, cv, SQLiteDatabase.CONFLICT_IGNORE)
    }
    fun item(id: Long): Item? = readableDatabase.rawQuery("SELECT $itemCols FROM items WHERE id=?", arrayOf("$id")).use { if (it.moveToFirst()) item(it) else null }
    fun items(collection: String? = null, modality: Modality? = null): List<Item> {
        val w = ArrayList<String>(); val a = ArrayList<String>()
        collection?.let { w += "collection=?"; a += it }; modality?.let { w += "modality=?"; a += it.name }
        val sql = "SELECT $itemCols FROM items" + (if (w.isEmpty()) "" else " WHERE " + w.joinToString(" AND ")) + " ORDER BY added_at DESC"
        return readableDatabase.rawQuery(sql, a.toTypedArray()).use { c -> generateSequence { if (c.moveToNext()) item(c) else null }.toList() }
    }
    fun nextQueued(): Item? = readableDatabase.rawQuery("SELECT $itemCols FROM items WHERE state='QUEUED' ORDER BY id LIMIT 1", null).use { if (it.moveToFirst()) item(it) else null }
    fun setState(id: Long, s: State, err: String? = null) {
        writableDatabase.execSQL("UPDATE items SET state=?, error=? WHERE id=?", arrayOf(s.name, err, id))
    }
    /** Interrupted indexing recovery: anything left PROCESSING goes back to the queue (its partial chunks are dropped). */
    fun recoverInterrupted() {
        writableDatabase.execSQL("DELETE FROM chunks WHERE item_id IN (SELECT id FROM items WHERE state='PROCESSING')")
        writableDatabase.execSQL("UPDATE items SET state='QUEUED' WHERE state='PROCESSING'")
    }
    fun requeueFailed() = writableDatabase.execSQL("UPDATE items SET state='QUEUED', error=NULL WHERE state='FAILED'")
    fun counts(): Map<String, Int> = readableDatabase.rawQuery("SELECT modality, COUNT(*) FROM items GROUP BY modality", null).use { c ->
        buildMap { while (c.moveToNext()) put(c.getString(0), c.getInt(1)) } }
    fun queueStats(): Pair<Int, Int> = readableDatabase.rawQuery("SELECT SUM(state='DONE'), COUNT(*) FROM items", null).use { it.moveToFirst(); it.getInt(0) to it.getInt(1) }

    fun addChunk(itemId: Long, ord: Int, label: String?, text: String?, ls: Int, le: Int, vec: FloatArray): Long {
        val cv = ContentValues().apply { put("item_id", itemId); put("ord", ord); put("label", label); put("text", text)
            put("line_start", ls); put("line_end", le); put("dims", vec.size); put("vec", VectorMath.toBytes(vec)) }
        return writableDatabase.insert("chunks", null, cv)
    }
    private fun chunk(c: Cursor) = Chunk(c.getLong(0), c.getLong(1), c.getInt(2), c.getString(3), c.getString(4), c.getInt(5), c.getInt(6), c.getInt(7))
    private val chunkCols = "id,item_id,ord,label,text,line_start,line_end,dims"
    fun chunk(id: Long): Chunk? = readableDatabase.rawQuery("SELECT $chunkCols FROM chunks WHERE id=?", arrayOf("$id")).use { if (it.moveToFirst()) chunk(it) else null }
    fun chunks(itemId: Long): List<Chunk> = readableDatabase.rawQuery("SELECT $chunkCols FROM chunks WHERE item_id=? ORDER BY ord", arrayOf("$itemId")).use { c -> generateSequence { if (c.moveToNext()) chunk(c) else null }.toList() }
    fun vector(chunkId: Long): FloatArray? = readableDatabase.rawQuery("SELECT vec FROM chunks WHERE id=?", arrayOf("$chunkId")).use { if (it.moveToFirst()) VectorMath.fromBytes(it.getBlob(0)) else null }

    /** Loads every vector with the given dimensionality into index entries. */
    fun loadEntries(dims: Int): List<IndexEntry> = readableDatabase.rawQuery(
        "SELECT c.id, c.item_id, i.modality, i.collection, c.vec FROM chunks c JOIN items i ON i.id=c.item_id WHERE c.dims=? AND i.state='DONE'", arrayOf("$dims")).use { c ->
        generateSequence { if (c.moveToNext()) IndexEntry(c.getLong(0), c.getLong(1), Modality.valueOf(c.getString(2)), c.getString(3), VectorMath.fromBytes(c.getBlob(4))) else null }.toList() }

    fun deleteItem(id: Long) = writableDatabase.execSQL("DELETE FROM items WHERE id=?", arrayOf(id))
    /** Drop all vectors and queue everything again (used when dimensionality or model changes). Notes keep their text in items.body. */
    fun clearEmbeddings() { writableDatabase.execSQL("DELETE FROM chunks"); writableDatabase.execSQL("UPDATE items SET state='QUEUED', error=NULL") }
    fun clearIndexedData() { writableDatabase.execSQL("DELETE FROM items"); writableDatabase.execSQL("DELETE FROM chunks") }
    fun clearAll() { clearIndexedData(); writableDatabase.execSQL("DELETE FROM history"); writableDatabase.execSQL("DELETE FROM collections WHERE name<>'Inbox'") }

    fun collections(): List<String> = readableDatabase.rawQuery("SELECT name FROM collections ORDER BY name", null).use { c -> generateSequence { if (c.moveToNext()) c.getString(0) else null }.toList() }
    fun addCollection(n: String) { writableDatabase.execSQL("INSERT OR IGNORE INTO collections VALUES(?)", arrayOf(n.trim())) }

    fun addHistory(q: String) { writableDatabase.execSQL("DELETE FROM history WHERE q=?", arrayOf(q)); writableDatabase.execSQL("INSERT INTO history(q,ts) VALUES(?,?)", arrayOf(q, System.currentTimeMillis())) }
    fun history(n: Int = 8): List<String> = readableDatabase.rawQuery("SELECT q FROM history ORDER BY ts DESC LIMIT $n", null).use { c -> generateSequence { if (c.moveToNext()) c.getString(0) else null }.toList() }
    fun clearHistory() = writableDatabase.execSQL("DELETE FROM history")
}
