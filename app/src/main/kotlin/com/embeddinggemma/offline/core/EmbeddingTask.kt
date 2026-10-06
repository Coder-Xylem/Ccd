package com.embeddinggemma.offline.core

/**
 * Task instructions as documented in the EmbeddingGemma 2 model card
 * (https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2).
 * Asymmetric tasks use a query form and a document form; symmetric tasks use
 * the query form for every input.
 */
enum class EmbeddingTask(val label: String, val promptName: String, private val taskText: String, val asymmetric: Boolean) {
    SEARCH("Search", "SearchQuery", "search result", true),
    QUESTION_ANSWERING("Question answering", "QuestionAnswering", "question answering", true),
    FACT_CHECKING("Fact checking", "FactChecking", "fact checking", true),
    CODE_RETRIEVAL("Code retrieval", "CodeRetrieval", "code retrieval", true),
    CLASSIFICATION("Classification", "Classification", "classification", false),
    CLUSTERING("Clustering", "Clustering", "clustering", false),
    SENTENCE_SIMILARITY("Sentence similarity", "SentenceSimilarity", "sentence similarity", false);

    /** `task: <task> | query: <content>` */
    fun formatQuery(content: String): String = "task: $taskText | query: $content"

    /** Asymmetric: `title: <title|none> | text: <content>`. Symmetric: same as the query form. */
    fun formatDocument(content: String, title: String? = null): String =
        if (asymmetric) "title: ${title?.takeIf { it.isNotBlank() } ?: "none"} | text: $content" else formatQuery(content)
}

enum class Modality { TEXT, IMAGE, AUDIO, VIDEO, CODE, NOTE }
