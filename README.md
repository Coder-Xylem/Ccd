# EmbeddingGemma Offline

A private, offline semantic search app for Android built on Google's **EmbeddingGemma 2** embedding model. Index your own text, notes, code, images and video; search, compare, classify, cluster and de-duplicate them on the phone. No account, no cloud, no `INTERNET` permission.

> **Build status: not verified.** This project was written in an environment with no Android SDK and no access to Google's Maven/Gradle hosts. The pure-Kotlin core (`core/`) was compiled and its unit tests pass. The Android layer (`data/`, `ui/`, Gradle, CI) has **not** been compiled. Expect to fix a few compile errors on first build, most likely in `data/Engines.kt`, which calls the MediaPipe Universal Embedder API exactly as Google's Android guide shows it. See [Known gaps](#known-gaps).

## Features
- Global semantic search with modality filters, collection filter, similarity floor, optional keyword boost (hybrid), search history, "Find similar" on any result
- **Ask my files**: retrieves the most relevant passages with source, section and line range. It does not generate answers
- Text, Markdown, source code (function/class-aware chunks with line ranges), notes, images, video (sampled frames, Fast 4 / Balanced 8 / Detailed 16)
- Cross-modal search through the shared embedding space: text→image, image→image, image→text/notes, text→video moments (needs a vision-capable bundle)
- Tools: Compare (text/text, text/image, image/image), zero-shot Classify, Organize (k-means clustering), Duplicates (never auto-deletes), Embedding Inspector, local benchmark
- Persistent index queue (SQLite) with pause / resume / cancel; interrupted items are re-queued on next start
- Collections, privacy dashboard, delete embeddings / index / all data

## Architecture
```
core/   pure Kotlin, no Android: EmbeddingTask, VectorMath (normalise, Matryoshka truncate), VectorIndex (exact cosine),
        Chunker, KMeans, Duplicates, ZeroShot, Hybrid, Embedder, EmbeddingEngine, MockEmbeddingEngine
data/   Store (SQLite), EmbeddingGemma2Engine (MediaPipe UniversalEmbedder), Indexer (queue worker), SemanticSearchEngine
ui/     Jetpack Compose screens: Search, Library, Tools, Settings
```
`EmbeddingEngine` has two implementations: `EmbeddingGemma2Engine` (real model, used by the app) and `MockEmbeddingEngine` (tests only; the app never falls back to it).

## Model setup
1. Download a `.litertlm` bundle from the Hugging Face **LiteRT Community**: `embeddinggemma-2-text-270m-litert-lm` (text, 270M), `embeddinggemma-2-text-vision-440m-litert-lm` (+images/video), or `embeddinggemma-2-740m-litert-lm` (omnimodal).
2. Settings → Model Manager → Import model. The file is copied into app-private storage; the variant is detected from the file name.
3. Read the model card there for license and acceptable-use terms. Weights are never committed to this repo (`*.litertlm` is git-ignored). Google's own pages are not fully consistent on the license (Apache 2.0 vs Gemma license), so check the card for the exact bundle you use.

## Matryoshka dimensions
The model outputs 768 values. This app keeps the first N (768 / 512 / 256 / 128) and **re-normalises**, since slicing alone breaks unit length. Query and stored vectors always share one size; changing it clears vectors and re-queues everything. Prompts follow the model card: queries `task: <task> | query: <text>`, documents `title: <title|none> | text: <text>`, symmetric tasks (classification, clustering, similarity) use the query form on both sides.

## Vector database
SQLite: `items` (also the queue), `chunks` (one row per passage/frame; float32 little-endian BLOB, dims, label, text, line range), `collections`, `history`. At startup vectors load into an in-memory exact cosine index (fine to roughly 100k vectors at 256 dims). No ANN index.

## Build
```
./gradlew testDebugUnitTest lintDebug assembleDebug
```
Needs JDK 17 and the Android SDK (platform 35). APK: `app/build/outputs/apk/debug/app-debug.apk`.

**GitHub Actions** (`.github/workflows/build.yml`): on every push it runs unit tests, lint, builds the debug APK and uploads it. Open the run under the repo's **Actions** tab → scroll to **Artifacts** → download `app-debug-apk`. Lint and test reports are in the `reports` artifact. Lint is configured not to fail the build (`abortOnError=false`); review the report.

## Device requirements
Android 8.0+ (API 26), 64-bit ARM. Google reports about 191 MB RAM for text-only and 567 MB for the full multimodal model on a Pixel 11 Pro; other devices vary. The Settings → Device check screen gives a rough recommendation.

## Known gaps
- **Not built or run on a device.** The MediaPipe calls (`UniversalEmbedder`, `Embedding.floatEmbedding()`) follow Google's guide of 2026-10-06 but are unverified; the dependency is `latest.release`, so pin a version once you have one that builds.
- **Audio is not supported** (no audio decoding or `AudioData` wiring), so no audio search; the omnimodal bundle's audio encoder is unused. **PDF is not supported** (no offline parser added).
- No UI/instrumented tests and no database tests (they need a device or Robolectric). Unit tests cover task formatting, normalisation, Matryoshka truncation, cosine, search, filtering, ranking, chunking, clustering, duplicates, zero-shot.
- Inference runs on CPU only. Interleaved text+image embedding exists in the engine (`embedTextAndImage`) but no screen uses it.
- Multilingual search is whatever the model provides; the app has no language-specific logic. Google notes quality varies by language.

## Privacy
All embedding, storage and search run on-device. The manifest removes `INTERNET`. Verify the built APK with `aapt dump permissions app-debug.apk`.

## Attribution
EmbeddingGemma 2 is a Google DeepMind model. This is an independent app, not affiliated with Google. MediaPipe and LiteRT are Google open-source projects.
