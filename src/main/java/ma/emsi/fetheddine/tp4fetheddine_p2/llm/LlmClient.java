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
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Named;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Service d'accès centralisé au modèle de langage Gemini via LangChain4j.
 * Gère la mémoire de chat et le rôle système pour maintenir le contexte.
 * MODIFIÉ : Maintenant @ApplicationScoped pour initialiser le RAG une seule fois.
 */
@ApplicationScoped
@Named("llmClient")
public class LlmClient implements Serializable {

    private transient ChatMemory chatMemory;
    private transient Assistant assistant;
    private String systemRole;

    public LlmClient() {
        System.out.println("Initialisation de LlmClient (@ApplicationScoped)...");

        // Lecture de la clé API depuis les variables d'environnement
        String envKey = System.getenv("GEMINI_KEY");
        if (envKey == null || envKey.isBlank()) {
            throw new IllegalStateException(
                    "Erreur : variable d'environnement GEMINI_KEY absente ou vide."
            );
        }

        // --- 1. Initialisation des modèles ---
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(envKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .logRequests(true) // Ajout des logs
                .logResponses(true)
                .build();

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(envKey)
                .modelName("text-embedding-004")
                .build();

        // --- 2. Création du ContentRetriever (RAG) ---
        System.out.println("Démarrage de l'ingestion des documents pour le RAG...");
        ContentRetriever contentRetriever = createCombinedRetriever(embeddingModel);
        System.out.println("Ingestion terminée.");

        // --- 3. Création de l'Assistant ---
        chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever) // INJECTION DU RAG
                .build();
    }

    /**
     * Crée un ContentRetriever unique contenant les deux PDF.
     */
    private ContentRetriever createCombinedRetriever(EmbeddingModel embeddingModel) {
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);

        // Ingestion de rag.pdf
        Document ragPdf = loadDocumentFromResources("rag.pdf");
        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build()
                .ingest(ragPdf);

        System.out.println("Document 'rag.pdf' ingéré.");

        // Ingestion de threat_report.pdf
        Document threatReportPdf = loadDocumentFromResources("threat_report.pdf");
        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build()
                .ingest(threatReportPdf);

        System.out.println("Document 'threat_report.pdf' ingéré.");

        // Création du retriever
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3) // Récupérer les 3 segments les plus pertinents
                .minScore(0.6) // Seuil de similarité minimum
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