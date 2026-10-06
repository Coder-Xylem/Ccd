package com.embeddinggemma.offline

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.embeddinggemma.offline.ui.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val app = (application as OfflineApp).state
        setContent { QuietTheme { Root(app) } }
    }
}

private val Ink = Color(0xFF000000)
private val Paper = Color(0xFFFFFFFF)

/** Two colours only (black / white) with neutral grey tints for secondary text. Follows the system light/dark setting. */
@Composable
fun QuietTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val scheme = if (dark) darkColorScheme(primary = Paper, onPrimary = Ink, background = Ink, onBackground = Paper, surface = Ink, onSurface = Paper,
        surfaceVariant = Color(0xFF1A1A1A), onSurfaceVariant = Color(0xFFAAAAAA), outline = Color(0xFF444444), secondaryContainer = Color(0xFF2A2A2A), onSecondaryContainer = Paper)
    else lightColorScheme(primary = Ink, onPrimary = Paper, background = Paper, onBackground = Ink, surface = Paper, onSurface = Ink,
        surfaceVariant = Color(0xFFF3F3F3), onSurfaceVariant = Color(0xFF666666), outline = Color(0xFFBBBBBB), secondaryContainer = Color(0xFFE6E6E6), onSecondaryContainer = Ink)
    MaterialTheme(colorScheme = scheme, content = content)
}

@Composable
fun Root(app: AppState) {
    var tab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Search", "Library", "Tools", "Settings")
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.background) {
                tabs.forEachIndexed { i, t -> NavigationBarItem(selected = tab == i, onClick = { tab = i }, icon = {}, label = { Text(t) }, alwaysShowLabel = true) }
            }
        },
    ) { pad ->
        Box(Modifier.padding(pad).fillMaxSize().clipToBounds()) {
            when (tab) { 0 -> SearchScreen(app); 1 -> LibraryScreen(app); 2 -> ToolsScreen(app); else -> SettingsScreen(app) }
        }
    }
}

/** Small thumbnail for a content Uri, decoded off the main thread. */
@Composable
fun Thumb(uri: String, modifier: Modifier = Modifier.size(64.dp)) {
    val ctx = LocalContext.current
    val bmp by produceState<androidx.compose.ui.graphics.ImageBitmap?>(null, uri) {
        value = withContext(Dispatchers.IO) {
            runCatching {
                val u = Uri.parse(uri)
                val o = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
                ctx.contentResolver.openInputStream(u)?.use { BitmapFactory.decodeStream(it, null, o) }
                var s = 1; while (maxOf(o.outWidth, o.outHeight) / s > 256) s *= 2
                ctx.contentResolver.openInputStream(u)?.use { BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = s }) }?.asImageBitmap()
            }.getOrNull()
        }
    }
    bmp?.let { Image(it, null, modifier, contentScale = ContentScale.Crop) } ?: Spacer(modifier)
}
