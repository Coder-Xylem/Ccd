package com.embeddinggemma.offline.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.embeddinggemma.offline.AppState
import com.embeddinggemma.offline.core.EmbeddingTask
import com.embeddinggemma.offline.core.Modality
import com.embeddinggemma.offline.core.VectorMath

@Composable fun Title(t: String) = Text(t, style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(vertical = 8.dp))
@Composable fun Muted(t: String, modifier: Modifier = Modifier) = Text(t, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = modifier)
@Composable fun Rule() = HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.4f))

@Composable
fun Confirm(title: String, text: String, onYes: () -> Unit, onNo: () -> Unit) = AlertDialog(
    onDismissRequest = onNo, title = { Text(title) }, text = { Text(text) },
    confirmButton = { TextButton(onClick = onYes) { Text("Confirm") } }, dismissButton = { TextButton(onClick = onNo) { Text("Cancel") } })

@Composable
fun <T> Chooser(label: String, options: List<T>, selected: T, name: (T) -> String, onPick: (T) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        OutlinedButton(onClick = { open = true }) { Text("$label: ${name(selected)}") }
        DropdownMenu(open, { open = false }) { options.forEach { o -> DropdownMenuItem(text = { Text(name(o)) }, onClick = { onPick(o); open = false }) } }
    }
}

fun pct(x: Float) = "${(x * 100).toInt().coerceIn(-100, 100)}%"
fun Modality.nice() = when (this) { Modality.TEXT -> "Document"; Modality.IMAGE -> "Image"; Modality.AUDIO -> "Audio"; Modality.VIDEO -> "Video"; Modality.CODE -> "Code"; Modality.NOTE -> "Note" }
val allTasks = EmbeddingTask.values().toList()

/** One vector per item = normalised mean of its chunk vectors at the active dimensionality. */
fun itemVectors(app: AppState, collection: String?): Map<Long, FloatArray> {
    val d = app.dims.value
    return app.store.items(collection).filter { it.state == com.embeddinggemma.offline.data.State.DONE }.mapNotNull { i ->
        val vs = app.store.chunks(i.id).filter { it.dims == d }.mapNotNull { app.store.vector(it.id) }
        if (vs.isEmpty()) null else i.id to VectorMath.mean(vs)
    }.toMap()
}
