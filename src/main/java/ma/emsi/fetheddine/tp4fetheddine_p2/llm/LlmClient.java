package ma.emsi.fetheddine.tp4fetheddine_p2.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Named;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * MODIFIÉ ÉTAPE 3 : Ajout de la Recherche Web
 */
@ApplicationScoped
@Named("llmClient")
public class LlmClient implements Serializable {

    private transient ChatMemory chatMemory;
    private transient Assistant assistant;
    private String systemRole;

    public LlmClient() {
        System.out.println("Initialisation de LlmClient (@ApplicationScoped) avec Routage + Web...");

        // --- Clé Gemini ---
        String envKey = System.getenv("GEMINI_KEY");
        if (envKey == null || envKey.isBlank()) {
            throw new IllegalStateException(
                    "Erreur : variable d'environnement GEMINI_KEY absente ou vide."
            );
        }

        // --- Clé Tavily (Web) ---
        String tavilyKey = System.getenv("TAVILY_KEY");
        if (tavilyKey == null || tavilyKey.isBlank()) {
            System.out.println("Avertissement : TAVILY_KEY non définie. La recherche Web sera désactivée.");
        }

        // --- 1. Initialisation des modèles ---
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(envKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(envKey)
                .modelName("text-embedding-004")
                .build();

        // --- 2. Création des ContentRetrievers (PDFs) ---
        System.out.println("Démarrage de l'ingestion pour le routage...");
        ContentRetriever ragRetriever = createPdfRetriever("rag.pdf", embeddingModel);
        System.out.println("'rag.pdf' ingéré.");
        ContentRetriever threatRetriever = createPdfRetriever("threat_report.pdf", embeddingModel);
        System.out.println("'threat_report.pdf' ingéré.");

        // --- 3. Configuration du QueryRouter (Map) ---
        Map<ContentRetriever, String> retrieverMap = new HashMap<>();
        retrieverMap.put(ragRetriever, "Répond aux questions sur 'Retrieval-Augmented Generation' (RAG), LangChain4j, fine-tuning, et les bases de données vectorielles.");
        retrieverMap.put(threatRetriever, "Répond aux questions sur un rapport de menaces de cybersécurité (threat report), 'vibe hacking', malwares, ransomwares, et les activités de hackers (par ex: Corée du Nord, Chine).");

        // --- AJOUT : Retriever Web ---
        if (tavilyKey != null && !tavilyKey.isBlank()) {
            WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                    .apiKey(tavilyKey)
                    .build();

            ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                    .webSearchEngine(webSearchEngine)
                    .build();

            retrieverMap.put(webRetriever, "Répond aux questions sur l'actualité, les événements récents, la météo, ou des sujets généraux non couverts par les documents fournis.");
            System.out.println("Recherche Web (Tavily) initialisée et ajoutée au routeur.");
        }
        // -----------------------------

        // --- 4. Création du RetrievalAugmentor ---
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, retrieverMap);
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        System.out.println("Routage final configuré.");

        // --- 5. Création de l'Assistant ---
        chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor) // Utilise l'Augmentor
                .build();
    }

    /**
     * Crée un ContentRetriever pour un seul PDF.
     */
    private ContentRetriever createPdfRetriever(String resourceName, EmbeddingModel embeddingModel) {
        Document document = loadDocumentFromResources(resourceName);
        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build()
                .ingest(document);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.6)
                .build();
    }

    /**
     * Charge un document depuis le dossier "resources" de l'application.
     */
    private Document loadDocumentFromResources(String resourceName) {
        try {
            URL resourceUrl = getClass().getClassLoader().getResource(resourceName);
            if (resourceUrl == null) {
                throw new RuntimeException("Ressource non trouvée : " + resourceName);
            }

            Path path = Paths.get(resourceUrl.toURI());
            DocumentParser parser = new ApacheTikaDocumentParser();
            return FileSystemDocumentLoader.loadDocument(path, parser);

        } catch (URISyntaxException e) {
            throw new RuntimeException("Erreur lors de la conversion de l'URL en URI : " + resourceName, e);
        }
    }


    public void setSystemRole(String role) {
        this.systemRole = role;
        chatMemory.clear();

        if (role != null && !role.trim().isEmpty()) {
            chatMemory.add(SystemMessage.from(role));
        }
    }


    public String ask(String prompt) {
        return assistant.chat(prompt);
    }

    // === Getters ===

    public String getSystemRole() {
        return systemRole;
    }
}